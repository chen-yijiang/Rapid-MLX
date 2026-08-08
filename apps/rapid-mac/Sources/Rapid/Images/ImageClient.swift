import Foundation

/// One generated (or edited) image plus the seed that produced it, so the
/// gallery can label results and an edit can re-target a specific image.
struct GeneratedImage: Identifiable, Hashable, Sendable {
    let id = UUID()
    /// Decoded PNG bytes (from the API's ``b64_json``).
    let pngData: Data
    /// The prompt that produced this image — shown as the gallery caption.
    let prompt: String
    /// True when this came from ``/v1/images/edits`` (vs. generations).
    let isEdit: Bool
}

/// Errors surfaced to the Images tab. Mirrors ``ChatStreamError`` in shape:
/// a small, user-actionable enum rather than raw ``URLError``/decode noise.
enum ImageClientError: Error, LocalizedError {
    case notReady
    case http(status: Int, message: String?)
    case emptyResponse
    case transport(String)

    var errorDescription: String? {
        switch self {
        case .notReady:
            return "The image model isn't running yet."
        case let .http(status, message):
            return message ?? "Image request failed (HTTP \(status))."
        case .emptyResponse:
            return "The server returned no image."
        case let .transport(detail):
            return detail
        }
    }
}

/// HTTP client for the OpenAI-compatible image endpoints. Non-streaming
/// (unlike ``ChatStreamClient``): a single request/response per image batch,
/// so it uses ``session.data(for:)`` like ``ServerProfileFetcher`` rather
/// than the SSE byte loop.
///
/// Port and bearer are passed per call — the caller reads
/// ``ServerManager.activePort`` / ``activeBearer`` at request time (they can
/// change across a stop/start reload), never caching them.
struct ImageClient {
    /// Generous timeout: a cold diffusion run on a large model can take tens
    /// of seconds, and the first call also pays a lazy model load.
    static let requestTimeout: TimeInterval = 300

    static let sharedSession: URLSession = {
        let config = URLSessionConfiguration.ephemeral
        config.timeoutIntervalForRequest = requestTimeout
        config.timeoutIntervalForResource = requestTimeout
        return URLSession(configuration: config)
    }()

    var session: URLSession = ImageClient.sharedSession

    static func loopbackURL(port: Int) -> URL {
        URL(string: "http://127.0.0.1:\(port)")!
    }

    // MARK: - Wire types

    private struct GenerationBody: Encodable {
        let model: String
        let prompt: String
        let n: Int
        let size: String
        let response_format = "b64_json"
        let seed: Int?
    }

    private struct ImageResponse: Decodable {
        struct Item: Decodable { let b64_json: String? }
        let data: [Item]
    }

    private struct ErrorEnvelope: Decodable {
        struct Inner: Decodable { let message: String? }
        let error: Inner?
    }

    // MARK: - Generations

    /// ``POST /v1/images/generations`` — text→image.
    func generate(
        prompt: String,
        model: String,
        size: String,
        count: Int,
        seed: Int?,
        port: Int,
        bearer: String?
    ) async throws -> [GeneratedImage] {
        let url = Self.loopbackURL(port: port).appendingPathComponent("v1/images/generations")
        var req = URLRequest(url: url)
        req.httpMethod = "POST"
        req.timeoutInterval = Self.requestTimeout
        req.setValue("application/json", forHTTPHeaderField: "Content-Type")
        req.setValue("application/json", forHTTPHeaderField: "Accept")
        applyBearer(&req, bearer)
        req.httpBody = try JSONEncoder().encode(
            GenerationBody(model: model, prompt: prompt, n: count, size: size, seed: seed)
        )
        let images = try await sendAndDecode(req)
        return images.map { GeneratedImage(pngData: $0, prompt: prompt, isEdit: false) }
    }

    // MARK: - Edits

    /// ``POST /v1/images/edits`` — image + instruction → image. Multipart,
    /// built by hand (there is no shared multipart helper in the app).
    func edit(
        imagePNG: Data,
        prompt: String,
        model: String,
        size: String,
        count: Int,
        seed: Int?,
        port: Int,
        bearer: String?
    ) async throws -> [GeneratedImage] {
        let url = Self.loopbackURL(port: port).appendingPathComponent("v1/images/edits")
        var req = URLRequest(url: url)
        req.httpMethod = "POST"
        req.timeoutInterval = Self.requestTimeout
        req.setValue("application/json", forHTTPHeaderField: "Accept")
        applyBearer(&req, bearer)

        let boundary = "rapid-\(UUID().uuidString)"
        req.setValue(
            "multipart/form-data; boundary=\(boundary)",
            forHTTPHeaderField: "Content-Type"
        )
        var fields: [(String, String)] = [
            ("prompt", prompt),
            ("model", model),
            ("size", size),
            ("n", String(count)),
            ("response_format", "b64_json"),
        ]
        if let seed { fields.append(("seed", String(seed))) }
        req.httpBody = Self.multipartBody(
            boundary: boundary, fields: fields,
            fileField: "image", fileName: "input.png",
            fileMime: "image/png", fileData: imagePNG
        )
        let images = try await sendAndDecode(req)
        return images.map { GeneratedImage(pngData: $0, prompt: prompt, isEdit: true) }
    }

    // MARK: - Shared

    private func applyBearer(_ req: inout URLRequest, _ bearer: String?) {
        if let bearer, !bearer.isEmpty {
            req.setValue("Bearer \(bearer)", forHTTPHeaderField: "Authorization")
        }
    }

    /// Send, validate status, decode the ``{data:[{b64_json}]}`` envelope
    /// into raw PNG byte blobs.
    private func sendAndDecode(_ req: URLRequest) async throws -> [Data] {
        let data: Data
        let response: URLResponse
        do {
            (data, response) = try await session.data(for: req)
        } catch {
            throw ImageClientError.transport(error.localizedDescription)
        }
        guard let http = response as? HTTPURLResponse else {
            throw ImageClientError.transport("Malformed server response.")
        }
        guard (200...299).contains(http.statusCode) else {
            let message = (try? JSONDecoder().decode(ErrorEnvelope.self, from: data))?
                .error?.message
            throw ImageClientError.http(status: http.statusCode, message: message)
        }
        guard let decoded = try? JSONDecoder().decode(ImageResponse.self, from: data),
              !decoded.data.isEmpty else {
            throw ImageClientError.emptyResponse
        }
        let blobs = decoded.data.compactMap { item -> Data? in
            guard let b64 = item.b64_json else { return nil }
            return Data(base64Encoded: b64)
        }
        guard !blobs.isEmpty else { throw ImageClientError.emptyResponse }
        return blobs
    }

    /// Assemble a multipart/form-data body from text fields + one file part.
    static func multipartBody(
        boundary: String,
        fields: [(String, String)],
        fileField: String,
        fileName: String,
        fileMime: String,
        fileData: Data
    ) -> Data {
        var body = Data()
        let dashes = "--\(boundary)\r\n"
        for (name, value) in fields {
            body.append(Data(dashes.utf8))
            body.append(Data("Content-Disposition: form-data; name=\"\(name)\"\r\n\r\n".utf8))
            body.append(Data("\(value)\r\n".utf8))
        }
        body.append(Data(dashes.utf8))
        body.append(Data(
            "Content-Disposition: form-data; name=\"\(fileField)\"; filename=\"\(fileName)\"\r\n".utf8
        ))
        body.append(Data("Content-Type: \(fileMime)\r\n\r\n".utf8))
        body.append(fileData)
        body.append(Data("\r\n".utf8))
        body.append(Data("--\(boundary)--\r\n".utf8))
        return body
    }
}

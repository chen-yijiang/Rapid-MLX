import Foundation
import Observation

/// State + orchestration for the Images tab. Mirrors ``ChatViewModel``:
/// an ``@Observable`` store the view binds to, owning the image client and
/// the results, and reading ``ServerManager.activePort`` / ``activeBearer``
/// at request time (never caching — they change across a reload).
@MainActor
@Observable
final class ImageGenViewModel {
    /// Speed-vs-quality is the user's real choice; the checkpoint name is
    /// subtext. ``match`` picks the alias out of the catalog.
    enum Quality: String, CaseIterable, Identifiable {
        case fast, best
        var id: String { rawValue }
        var title: String { self == .fast ? "Fast" : "Best" }
        var symbol: String { self == .fast ? "bolt.fill" : "sparkles" }
        var subtitle: String { self == .fast ? "FLUX.2 Klein" : "Z-Image Turbo" }
        var etaHint: String { self == .fast ? "~10s" : "~33s" }
        /// Substring that identifies this tier's alias in the catalog.
        var match: String { self == .fast ? "klein" : "z-image" }
        /// Steps used to seed the progress bar before the server reports one.
        var seedSteps: Int { self == .fast ? 4 : 8 }
    }

    /// Aspect ratio as three friendly buttons rather than a "512×512" string.
    enum Aspect: String, CaseIterable, Identifiable {
        case square, portrait, landscape
        var id: String { rawValue }
        var label: String {
            switch self {
            case .square: return "1:1"
            case .portrait: return "3:4"
            case .landscape: return "4:3"
            }
        }
        var size: String {
            switch self {
            case .square: return "1024x1024"
            case .portrait: return "768x1024"
            case .landscape: return "1024x768"
            }
        }
    }

    /// The two phases of a render, shown very differently: a reassuring
    /// (indeterminate) cold-load, then a determinate denoise.
    enum Phase: Equatable { case preparing, denoising }

    /// A few one-tap prompt starters to beat the blank page.
    static let starters: [String] = [
        "A cozy ramen shop at night in the rain, neon, steam, 35mm",
        "Studio portrait of an elderly fisherman, dramatic side light",
        "A minimalist product shot of a ceramic mug on linen",
        "A whale drifting through clouds above a city at dusk",
    ]

    // MARK: - Composed input
    var prompt: String = ""
    var quality: Quality = .fast
    var aspect: Aspect = .square

    // MARK: - Catalog
    /// The alias each quality tier resolves to (from the ``[image:gen]`` rows).
    var imageModels: [ModelEntry] = []
    var catalogLoaded: Bool = false
    private(set) var selectedAlias: String = ""

    // MARK: - Results
    /// Newest-first session gallery (the filmstrip).
    var results: [GeneratedImage] = []
    /// The focal image the stage shows; nil ⇒ newest, or empty state.
    var activeID: GeneratedImage.ID?
    var activeImage: GeneratedImage? {
        if let activeID, let hit = results.first(where: { $0.id == activeID }) { return hit }
        return results.first
    }

    // MARK: - Run state
    var isGenerating: Bool = false
    var phase: Phase = .preparing
    var progress: ImageClient.ImageProgress?
    var errorMessage: String?
    /// True only for the window between "Cancel pressed" and the run ending.
    private(set) var cancelling: Bool = false
    /// When the current run started — drives a live elapsed clock in the HUD
    /// that keeps moving even during the cold model-load phase.
    private(set) var genStartedAt: Date?

    /// Steps the bar should assume before the server reports a live total.
    var estimatedSteps: Int { progress?.total ?? quality.seedSteps }

    // MARK: - Edit (parked lane, kept for later)
    var editSource: GeneratedImage?

    private let client = ImageClient()
    private let server: ServerManager

    init(server: ServerManager) {
        self.server = server
    }

    var canSubmit: Bool {
        !isGenerating
            && !prompt.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
            && !selectedAlias.isEmpty
    }

    /// Does a model for the chosen quality exist in the catalog?
    func hasModel(for quality: Quality) -> Bool {
        imageModels.contains { $0.alias.localizedCaseInsensitiveContains(quality.match) }
    }

    func setQuality(_ q: Quality) {
        quality = q
        resolveAlias()
    }

    func use(starter: String) {
        prompt = starter
    }

    func select(_ image: GeneratedImage) {
        activeID = image.id
        prompt = image.prompt
    }

    /// Load the image-gen alias catalog (safe to call repeatedly).
    func refreshCatalog() async {
        guard let binary = server.binaryPath else { return }
        imageModels = await ModelCatalog.imageEntries(binary: binary)
        catalogLoaded = true
        resolveAlias()
    }

    /// Point ``selectedAlias`` at the current quality tier, falling back to the
    /// other tier (then any image model) so Generate is never dead.
    private func resolveAlias() {
        let byTier = imageModels.first { $0.alias.localizedCaseInsensitiveContains(quality.match) }
        selectedAlias = (byTier ?? imageModels.first)?.alias ?? ""
    }

    // MARK: - Generate

    func submit() async {
        if let source = editSource {
            await runEdit(source: source)
        } else {
            await runGenerate()
        }
    }

    func cancel() {
        guard isGenerating, !cancelling else { return }
        cancelling = true
        let port = server.activePort
        let bearer = server.activeBearer
        Task { await client.cancel(port: port, bearer: bearer) }
    }

    private func runGenerate() async {
        let trimmed = prompt.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty, !selectedAlias.isEmpty else { return }
        await withRequest {
            let hf = self.imageModels.first { $0.alias == self.selectedAlias }?.hfRepo
            guard await self.server.ensureServing(alias: self.selectedAlias, hfPath: hf) else {
                throw ImageClientError.notReady
            }
            let port = self.server.activePort
            let bearer = self.server.activeBearer
            let poll = self.startPolling(port: port, bearer: bearer)
            defer { poll.cancel() }
            let images = try await self.client.generate(
                prompt: trimmed, model: self.selectedAlias, size: self.aspect.size,
                count: 1, seed: nil, port: port, bearer: bearer
            )
            if let first = images.first {
                self.results.insert(contentsOf: images, at: 0)
                self.activeID = first.id
            }
            // Empty (cancelled before the first image) leaves the gallery as-is.
        }
    }

    private func runEdit(source: GeneratedImage) async {
        let trimmed = prompt.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty, !selectedAlias.isEmpty else { return }
        await withRequest {
            let hf = self.imageModels.first { $0.alias == self.selectedAlias }?.hfRepo
            guard await self.server.ensureServing(alias: self.selectedAlias, hfPath: hf) else {
                throw ImageClientError.notReady
            }
            let port = self.server.activePort
            let bearer = self.server.activeBearer
            let poll = self.startPolling(port: port, bearer: bearer)
            defer { poll.cancel() }
            let images = try await self.client.edit(
                imagePNG: source.pngData, prompt: trimmed, model: self.selectedAlias,
                size: self.aspect.size, count: 1, seed: nil, port: port, bearer: bearer
            )
            if let first = images.first {
                self.results.insert(contentsOf: images, at: 0)
                self.activeID = first.id
            }
        }
    }

    /// Poll the server's live denoise progress ~3×/second and mirror it into
    /// ``progress`` / ``phase`` so the stage shows a true step bar and ETA.
    private func startPolling(port: Int, bearer: String?) -> Task<Void, Never> {
        Task { [weak self] in
            while !Task.isCancelled {
                if let snap = await self?.client.fetchProgress(port: port, bearer: bearer) {
                    self?.progress = snap
                    self?.phase = snap.running ? .denoising : .preparing
                }
                try? await Task.sleep(for: .milliseconds(300))
            }
        }
    }

    func beginEdit(_ image: GeneratedImage) {
        editSource = image
        prompt = ""
        errorMessage = nil
    }

    func cancelEdit() { editSource = nil }

    /// Shared request wrapper: flips run state, resets progress, and funnels
    /// every failure into ``errorMessage``.
    private func withRequest(_ body: @escaping () async throws -> Void) async {
        isGenerating = true
        cancelling = false
        phase = .preparing
        progress = nil
        genStartedAt = Date()
        errorMessage = nil
        defer {
            isGenerating = false
            cancelling = false
            progress = nil
            genStartedAt = nil
        }
        do {
            try await body()
            editSource = nil
            prompt = ""
        } catch let error as ImageClientError {
            errorMessage = error.errorDescription
        } catch {
            errorMessage = error.localizedDescription
        }
    }
}

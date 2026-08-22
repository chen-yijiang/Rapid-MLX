import Foundation
import Testing
@testable import Rapid

@MainActor
@Suite("Multi-model residency")
struct ModelResidencyTests {
    @Test("Residency status decodes the server wire format")
    func decodesSnapshot() throws {
        let data = Data(
            #"""
            {
              "memory_limit_bytes": 34359738368,
              "memory_used_bytes": 10737418240,
              "memory_available_bytes": 23622320128,
              "idle_ttl_seconds": 1800,
              "loads_total": 2,
              "evictions_total": 1,
              "models": [{
                "id": "flux2-klein-4b",
                "model_path": "Runware/FLUX.2-klein-4B",
                "aliases": ["flux-klein"],
                "modality": "image-gen",
                "state": "resident",
                "pinned": false,
                "primary": false,
                "active_requests": 0,
                "estimated_bytes": 6335076761,
                "measured_bytes": 5905580032,
                "idle_seconds": 12.5,
                "performance": {
                  "kv_cache_turboquant": "k8v4",
                  "prefix_cache_enabled": true,
                  "cache_memory_mb": 4096
                }
              }]
            }
            """#.utf8
        )

        let snapshot = try JSONDecoder().decode(ModelResidencySnapshot.self, from: data)

        #expect(snapshot.memoryLimitBytes == 34_359_738_368)
        #expect(snapshot.memoryUsedBytes == 10_737_418_240)
        #expect(snapshot.loadsTotal == 2)
        #expect(snapshot.evictionsTotal == 1)
        #expect(snapshot.models.first?.modality == "image-gen")
        #expect(snapshot.models.first?.displayBytes == 6_335_076_761)
        #expect(snapshot.contains("flux2-klein-4b"))
        #expect(snapshot.contains("flux-klein"))
        #expect(snapshot.contains("Runware/FLUX.2-klein-4B"))
        #expect(snapshot.models.first?.performance == ResidentPerformanceStatus(
            config: ModelPerfConfig(
                kvCacheMode: .turboquantK8V4,
                prefixCacheEnabled: true,
                cacheMemoryMB: 4096
            )
        ))
    }

    @Test("Resident rows prefer the catalog alias over the HF path")
    func residentDisplayName() {
        let status = ResidentModelStatus(
            id: "mlx-community/Qwen3.5-4B-MLX-4bit",
            modelPath: "mlx-community/Qwen3.5-4B-MLX-4bit",
            aliases: ["qwen3.5-4b-4bit"],
            modality: "text",
            state: "resident",
            pinned: true,
            primary: true,
            activeRequests: 0,
            estimatedBytes: 1,
            measuredBytes: nil,
            idleSeconds: 0
        )

        #expect(status.displayName() == "qwen3.5-4b-4bit")
        #expect(status.displayName(preferredAlias: "qwen3.5-4b-4bit") == "qwen3.5-4b-4bit")
    }

    @Test("Active request count ignores evicting engines and invalid negative values")
    func activeRequestCount() {
        func status(_ id: String, state: String, active: Int) -> ResidentModelStatus {
            ResidentModelStatus(
                id: id,
                modelPath: id,
                aliases: [],
                modality: "text",
                state: state,
                pinned: false,
                primary: false,
                activeRequests: active,
                estimatedBytes: 1,
                measuredBytes: nil,
                idleSeconds: 0
            )
        }
        let snapshot = ModelResidencySnapshot(
            memoryLimitBytes: 1,
            memoryUsedBytes: 1,
            memoryAvailableBytes: 0,
            idleTTLSeconds: 1,
            loadsTotal: 1,
            evictionsTotal: 0,
            models: [
                status("primary", state: "resident", active: 2),
                status("secondary", state: "loading", active: 1),
                status("leaving", state: "evicting", active: 9),
                status("invalid", state: "resident", active: -4),
            ]
        )

        #expect(snapshot.activeRequestCount == 3)
    }

    @Test("Assistant replacement counts only text and VLM requests")
    func assistantReplacementRequestCount() {
        let snapshot = requestSnapshot(textRequests: 2, imageRequests: 5)
        #expect(snapshot.activeRequestCount == 7)
        #expect(snapshot.activeRequestCount(replacingGroup: .assistant) == 2)
    }

    @Test("Active-request switch confirmations resolve in FIFO order")
    func activeRequestSwitchConfirmationFIFO() {
        var queue = ActiveRequestSwitchConfirmationQueue()
        let firstRequest = UUID()
        let secondRequest = UUID()
        let first = ActiveRequestSwitchWarning(
            id: UUID(), currentAlias: "a", targetAlias: "b", activeRequests: 1
        )
        let second = ActiveRequestSwitchWarning(
            id: UUID(), currentAlias: "a", targetAlias: "c", activeRequests: 2
        )

        queue.enqueue(first, requestID: firstRequest)
        queue.enqueue(second, requestID: secondRequest)
        #expect(queue.currentWarning == first)

        queue.resolveCurrent(second, confirmed: true)
        #expect(queue.currentWarning == first)
        queue.resolveCurrent(first, confirmed: false)
        #expect(queue.takeDecision(for: firstRequest) == false)
        #expect(queue.currentWarning == second)
        queue.resolveCurrent(second, confirmed: true)
        #expect(queue.takeDecision(for: secondRequest) == true)
        #expect(queue.currentWarning == nil)
    }

    @Test("A failed decision-point refresh fails safe and cancellation is distinct")
    func failedRefreshPromptsAndReturnsCancellation() async throws {
        let configuration = URLSessionConfiguration.ephemeral
        configuration.protocolClasses = [ResidencyFetchFailureProtocol.self]
        var client = ServerResidencyClient()
        client.session = URLSession(configuration: configuration)
        let server = ServerManager(testingState: .ready(alias: "current"))
        server._testSetResidencyClient(client)
        server._testInstallChild(ProcessGroupChild.testStub())

        let switchTask = Task { @MainActor in
            await server.ensureServingOutcome(
                alias: "target",
                hfPath: nil,
                estimatedMemoryGB: nil,
                residencyEligible: false
            )
        }
        for _ in 0..<100 where server.pendingActiveRequestSwitch == nil {
            try await Task.sleep(for: .milliseconds(10))
        }
        let warning = try #require(server.pendingActiveRequestSwitch)
        #expect(warning.activeRequests == nil)

        server.cancelActiveRequestSwitch(warning)
        #expect(await switchTask.value == .cancelled)
        #expect(server.state == .ready(alias: "current"))
    }

    @Test("A current active request can be confirmed before process replacement")
    func activeRequestConfirmationContinuesSwitch() async throws {
        let server = makeSwitchServer(protocolClass: ActiveResidencyProtocol.self)
        let switchTask = Task { @MainActor in
            await server.ensureServingOutcome(
                alias: "target", hfPath: nil, estimatedMemoryGB: nil,
                residencyEligible: false
            )
        }
        for _ in 0..<100 where server.pendingActiveRequestSwitch == nil {
            try await Task.sleep(for: .milliseconds(10))
        }
        let warning = try #require(server.pendingActiveRequestSwitch)
        #expect(warning.activeRequests == 2)
        server.confirmActiveRequestSwitch(warning)

        // There is intentionally no binary in this harness, so reaching a
        // normal startup failure proves confirmation continued past the gate.
        #expect(await switchTask.value == .failed)
        #expect(server.pendingActiveRequestSwitch == nil)
    }

    @Test("A current zero-request snapshot does not prompt")
    func zeroActiveRequestsDoNotPrompt() async {
        let server = makeSwitchServer(protocolClass: IdleResidencyProtocol.self)
        let outcome = await server.ensureServingOutcome(
            alias: "target", hfPath: nil, estimatedMemoryGB: nil,
            residencyEligible: false
        )
        #expect(outcome == .failed)
        #expect(server.pendingActiveRequestSwitch == nil)
    }

    @Test("Restart confirms before stopping an active server")
    func restartCancellationPreservesServer() async throws {
        let server = makeSwitchServer(protocolClass: ActiveResidencyProtocol.self)
        let restartTask = Task { @MainActor in
            await server.restartServingOutcome(alias: "current", hfPath: nil)
        }
        for _ in 0..<100 where server.pendingActiveRequestSwitch == nil {
            try await Task.sleep(for: .milliseconds(10))
        }
        let warning = try #require(server.pendingActiveRequestSwitch)
        server.cancelActiveRequestSwitch(warning)

        #expect(await restartTask.value == .cancelled)
        #expect(server.state == .ready(alias: "current"))
    }

    @Test("Speculative restart cancellation preserves the active server")
    func speculativeRestartCancellationPreservesServer() async throws {
        let server = makeSwitchServer(protocolClass: ActiveResidencyProtocol.self)
        let restartTask = Task { @MainActor in
            await server.restartForSpeculativePerformance(alias: "current")
        }
        for _ in 0..<100 where server.pendingActiveRequestSwitch == nil {
            try await Task.sleep(for: .milliseconds(10))
        }
        let warning = try #require(server.pendingActiveRequestSwitch)
        server.cancelActiveRequestSwitch(warning)

        #expect(await restartTask.value == .cancelled)
        #expect(server.state == .ready(alias: "current"))
    }

    @Test("Overlapping switches revalidate after the preceding decision")
    func overlappingSwitchesRevalidate() async throws {
        OverlappingResidencyProtocol.fetchCount = 0
        OverlappingResidencyProtocol.lastTimeout = nil
        let server = makeSwitchServer(protocolClass: OverlappingResidencyProtocol.self)
        let firstTask = Task { @MainActor in
            await server.ensureServingOutcome(
                alias: "first", hfPath: nil, estimatedMemoryGB: nil,
                residencyEligible: false
            )
        }
        let secondTask = Task { @MainActor in
            await server.ensureServingOutcome(
                alias: "second", hfPath: nil, estimatedMemoryGB: nil,
                residencyEligible: false
            )
        }

        for _ in 0..<100 where server.pendingActiveRequestSwitch == nil {
            try await Task.sleep(for: .milliseconds(10))
        }
        let firstWarning = try #require(server.pendingActiveRequestSwitch)
        #expect(OverlappingResidencyProtocol.fetchCount == 1)
        #expect(OverlappingResidencyProtocol.lastTimeout == 2)
        // Stand in for the first transaction installing B while it still owns
        // the operation gate; the second transaction must observe B afterward.
        server._testSetState(.ready(alias: "replacement"))
        server.cancelActiveRequestSwitch(firstWarning)
        #expect(await firstTask.value == .cancelled)

        // The queued operation fetches again and describes the current process,
        // rather than reusing the first transaction's A-era warning.
        for _ in 0..<100 where server.pendingActiveRequestSwitch == nil {
            try await Task.sleep(for: .milliseconds(10))
        }
        let secondWarning = try #require(server.pendingActiveRequestSwitch)
        #expect(secondWarning.currentAlias == "replacement")
        #expect(OverlappingResidencyProtocol.fetchCount == 2)
        server.cancelActiveRequestSwitch(secondWarning)
        #expect(await secondTask.value == .cancelled)
    }

    @Test("Production restart surfaces use the confirmation-aware manager API")
    func restartSurfacesDoNotStopDirectly() throws {
        let rapidMacRoot = URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent()
            .deletingLastPathComponent()
            .deletingLastPathComponent()
        for relativePath in [
            "Sources/Rapid/UI/AudioView.swift",
            "Sources/Rapid/UI/ImagesView.swift",
            "Sources/Rapid/UI/SettingsConnectorsPanel.swift",
        ] {
            let source = try String(
                contentsOf: rapidMacRoot.appendingPathComponent(relativePath),
                encoding: .utf8
            )
            #expect(!source.contains("await server.stop()"), "Direct stop bypass in \(relativePath)")
            #expect(source.contains("restartServingOutcome"))
        }
    }

    @Test("Connector restart prefers a resident text model over the process-owning audio alias")
    func connectorRestartTextAlias() {
        let text = ResidentModelStatus(
            id: "qwen3.5-4b-4bit",
            modelPath: "mlx-community/Qwen3.5-4B-MLX-4bit",
            aliases: [],
            modality: "text",
            state: "resident",
            pinned: false,
            primary: false,
            activeRequests: 0,
            estimatedBytes: 1,
            measuredBytes: nil,
            idleSeconds: 0
        )
        let snapshot = ModelResidencySnapshot(
            memoryLimitBytes: 1,
            memoryUsedBytes: 1,
            memoryAvailableBytes: 0,
            idleTTLSeconds: 1,
            loadsTotal: 1,
            evictionsTotal: 0,
            models: [text]
        )

        #expect(snapshot.preferredTextAlias(fallback: "qwen3-tts-4bit") == "qwen3.5-4b-4bit")
        #expect(ModelResidencySnapshot.empty.preferredTextAlias(fallback: "legacy-chat") == "legacy-chat")
    }

    @Test("Server readiness is resolved for every resident alias")
    func aliasSpecificReadiness() {
        let image = ResidentModelStatus(
            id: "flux2-klein-4b",
            modelPath: "Runware/FLUX.2-klein-4B",
            aliases: ["flux-klein"],
            modality: "image-gen",
            state: "resident",
            pinned: false,
            primary: false,
            activeRequests: 0,
            estimatedBytes: 6_335_076_761,
            measuredBytes: nil,
            idleSeconds: 0
        )
        let snapshot = ModelResidencySnapshot(
            memoryLimitBytes: 25 * 1_073_741_824,
            memoryUsedBytes: 10 * 1_073_741_824,
            memoryAvailableBytes: 15 * 1_073_741_824,
            idleTTLSeconds: 1800,
            loadsTotal: 1,
            evictionsTotal: 0,
            models: [image]
        )
        let server = ServerManager(
            testingState: .ready(alias: "qwen3.5-4b-4bit"),
            residency: snapshot
        )

        #expect(server.isModelResident("qwen3.5-4b-4bit"))
        #expect(server.isModelResident("flux2-klein-4b"))
        #expect(server.isModelResident("flux-klein"))
        #expect(!server.isModelResident("z-image-turbo"))

        guard case .ready(let alias) = server.readinessState(for: "flux-klein") else {
            Issue.record("Expected resident image alias to be ready")
            return
        }
        #expect(alias == "flux-klein")
    }

    @Test("Resident ceiling reuses the Mac usable-RAM bucket")
    func residentMemoryCeiling() {
        #expect(ModelSizing.residentMemoryCeilingGB(on: mockMac(ramGB: 32)) == 25)
        #expect(ModelSizing.residentMemoryCeilingGB(on: mockMac(ramGB: 8)) == 6)
        #expect(ModelSizing.residentMemoryCeilingGB(on: mockMac(ramGB: 4)) == 4)
    }

    @Test("Residency load sends typed performance config and reload intent")
    func loadRequestCarriesPerformance() async throws {
        ResidencyLoadCaptureProtocol.capturedBody = nil
        let configuration = URLSessionConfiguration.ephemeral
        configuration.protocolClasses = [ResidencyLoadCaptureProtocol.self]
        var client = ServerResidencyClient()
        client.session = URLSession(configuration: configuration)

        let result = await client.load(
            alias: "qwen3.5-4b-4bit",
            hfPath: "mlx-community/Qwen3.5-4B-MLX-4bit",
            estimatedSizeGB: 4,
            imageMode: .editing,
            performance: ModelPerfConfig(
                kvCacheMode: .turboquantK8V4,
                prefixCacheEnabled: false,
                cacheMemoryMB: 4096
            ),
            reloadIfChanged: true,
            port: 8000,
            bearer: "secret"
        )
        guard case .loaded = result else {
            Issue.record("Expected the stubbed residency load to succeed")
            return
        }
        let body = try #require(ResidencyLoadCaptureProtocol.capturedBody)
        let json = try #require(JSONSerialization.jsonObject(with: body) as? [String: Any])
        let performance = try #require(json["performance"] as? [String: Any])
        #expect(json["reload_if_changed"] as? Bool == true)
        #expect(json["image_mode"] as? String == "editing")
        #expect(performance["kv_cache_dtype"] == nil)
        #expect(performance["kv_cache_turboquant"] as? String == "k8v4")
        #expect(performance["prefix_cache_enabled"] as? Bool == false)
        #expect(performance["cache_memory_mb"] as? Int == 4096)
    }

    @Test("Image residency estimate uses catalog bytes plus runtime margin")
    func imageEstimateUsesDownloadSize() {
        let estimate = ModelSizing.residentEstimateGB(
            alias: "z-image-turbo",
            sizeText: "5.5 GiB"
        )
        #expect(abs(estimate - 7.375) < 0.001)
        #expect(estimate > ModelSizing.residentEstimateGB(alias: "z-image-turbo"))
    }

    @Test("Selecting a chat model does not retain the legacy server restart flow")
    func selectionDoesNotRestartServer() throws {
        let rapidMacRoot = URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent()
            .deletingLastPathComponent()
            .deletingLastPathComponent()
        let sourceURL = rapidMacRoot
            .appendingPathComponent("Sources/Rapid/UI/ContentView.swift")
        let source = try String(contentsOf: sourceURL, encoding: .utf8)

        #expect(!source.contains("pendingReloadAlias"))
        #expect(!source.contains("Switch and reload"))
        #expect(!source.contains("Stops the current model and loads"))
    }

    @Test("Selecting a cached chat model activates it immediately")
    func cachedChatSelectionActivates() {
        #expect(ContentView.activatesChatModelOnSelection(
            isResident: false,
            isCached: true
        ))
        #expect(ContentView.activatesChatModelOnSelection(
            isResident: true,
            isCached: false
        ))
        #expect(!ContentView.activatesChatModelOnSelection(
            isResident: false,
            isCached: false
        ))
    }

    private func mockMac(ramGB: Int) -> MacHardware {
        MacHardware(
            brandString: "Apple M3 Pro",
            family: .m3,
            tier: .pro,
            physicalRAMBytes: UInt64(ramGB) * UInt64(1 << 30),
            memoryBandwidthGBs: 150
        )
    }

    private func makeSwitchServer(protocolClass: AnyClass) -> ServerManager {
        let configuration = URLSessionConfiguration.ephemeral
        configuration.protocolClasses = [protocolClass]
        var client = ServerResidencyClient()
        client.session = URLSession(configuration: configuration)
        let server = ServerManager(testingState: .ready(alias: "current"))
        server._testSetResidencyClient(client)
        server._testInstallChild(ProcessGroupChild.testStub())
        return server
    }

    private func requestSnapshot(textRequests: Int, imageRequests: Int) -> ModelResidencySnapshot {
        func status(_ id: String, modality: String, requests: Int) -> ResidentModelStatus {
            ResidentModelStatus(
                id: id, modelPath: id, aliases: [], modality: modality,
                state: "resident", pinned: false, primary: modality == "text",
                activeRequests: requests, estimatedBytes: 1,
                measuredBytes: nil, idleSeconds: 0
            )
        }
        return ModelResidencySnapshot(
            memoryLimitBytes: 1, memoryUsedBytes: 1, memoryAvailableBytes: 0,
            idleTTLSeconds: 1, loadsTotal: 1, evictionsTotal: 0,
            models: [
                status("text", modality: "text", requests: textRequests),
                status("image", modality: "image-gen", requests: imageRequests),
            ]
        )
    }
}

private final class ResidencyFetchFailureProtocol: URLProtocol, @unchecked Sendable {
    override class func canInit(with request: URLRequest) -> Bool { true }
    override class func canonicalRequest(for request: URLRequest) -> URLRequest { request }

    override func startLoading() {
        client?.urlProtocol(self, didFailWithError: URLError(.timedOut))
    }

    override func stopLoading() {}
}

private class ResidencySnapshotProtocol: URLProtocol, @unchecked Sendable {
    class var activeRequests: Int { 0 }
    override class func canInit(with request: URLRequest) -> Bool { true }
    override class func canonicalRequest(for request: URLRequest) -> URLRequest { request }

    override func startLoading() {
        let payload = #"{"memory_limit_bytes":1,"memory_used_bytes":1,"memory_available_bytes":0,"idle_ttl_seconds":1,"loads_total":1,"evictions_total":0,"models":[{"id":"current","model_path":"current","aliases":[],"modality":"text","state":"resident","pinned":true,"primary":true,"active_requests":\#(Self.activeRequests),"estimated_bytes":1,"measured_bytes":null,"idle_seconds":0}]}"#.data(using: .utf8)!
        let response = HTTPURLResponse(
            url: request.url!, statusCode: 200, httpVersion: "HTTP/1.1", headerFields: nil
        )!
        client?.urlProtocol(self, didReceive: response, cacheStoragePolicy: .notAllowed)
        client?.urlProtocol(self, didLoad: payload)
        client?.urlProtocolDidFinishLoading(self)
    }

    override func stopLoading() {}
}

private final class ActiveResidencyProtocol: ResidencySnapshotProtocol, @unchecked Sendable {
    override class var activeRequests: Int { 2 }
}

private final class IdleResidencyProtocol: ResidencySnapshotProtocol, @unchecked Sendable {}

private final class OverlappingResidencyProtocol: ResidencySnapshotProtocol, @unchecked Sendable {
    nonisolated(unsafe) static var fetchCount = 0
    nonisolated(unsafe) static var lastTimeout: TimeInterval?
    override class var activeRequests: Int { 1 }

    override func startLoading() {
        Self.fetchCount += 1
        Self.lastTimeout = request.timeoutInterval
        super.startLoading()
    }
}

private final class ResidencyLoadCaptureProtocol: URLProtocol, @unchecked Sendable {
    nonisolated(unsafe) static var capturedBody: Data?

    override class func canInit(with request: URLRequest) -> Bool { true }
    override class func canonicalRequest(for request: URLRequest) -> URLRequest { request }

    override func startLoading() {
        Self.capturedBody = request.httpBody ?? Self.readBodyStream(request.httpBodyStream)
        let payload = #"{"id":"qwen3.5-4b-4bit","model_path":"mlx-community/Qwen3.5-4B-MLX-4bit","aliases":[],"modality":"text","state":"resident","pinned":true,"primary":true,"active_requests":0,"estimated_bytes":1,"measured_bytes":null,"idle_seconds":0,"performance":{"kv_cache_turboquant":"k8v4","prefix_cache_enabled":false,"cache_memory_mb":4096}}"#.data(using: .utf8)!
        let response = HTTPURLResponse(
            url: request.url!, statusCode: 200, httpVersion: "HTTP/1.1", headerFields: nil
        )!
        client?.urlProtocol(self, didReceive: response, cacheStoragePolicy: .notAllowed)
        client?.urlProtocol(self, didLoad: payload)
        client?.urlProtocolDidFinishLoading(self)
    }

    override func stopLoading() {}

    private static func readBodyStream(_ stream: InputStream?) -> Data? {
        guard let stream else { return nil }
        stream.open()
        defer { stream.close() }
        var data = Data()
        var buffer = [UInt8](repeating: 0, count: 4096)
        while true {
            let count = buffer.withUnsafeMutableBufferPointer { pointer in
                stream.read(pointer.baseAddress!, maxLength: pointer.count)
            }
            if count > 0 { data.append(buffer, count: count) }
            if count == 0 { return data }
            if count < 0 { return nil }
        }
    }
}

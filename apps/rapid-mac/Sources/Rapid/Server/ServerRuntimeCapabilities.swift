import Darwin
import Foundation

/// Feature switches supported by the selected `rapid-mlx serve` binary.
///
/// The desktop can run against a managed runtime override that survives app
/// source updates. Probe the actual CLI before spawn so a stale runtime does
/// not reject newer desktop-only flags during argument parsing.
struct ServerRuntimeCapabilities: Equatable, Sendable {
    var supportsResidentMemoryLimitGB: Bool
    var supportsResidentModelIdleTTL: Bool

    static let conservative = ServerRuntimeCapabilities(
        supportsResidentMemoryLimitGB: false,
        supportsResidentModelIdleTTL: false
    )

    static let allKnown = ServerRuntimeCapabilities(
        supportsResidentMemoryLimitGB: true,
        supportsResidentModelIdleTTL: true
    )

    static func parse(serveHelp text: String) -> ServerRuntimeCapabilities {
        ServerRuntimeCapabilities(
            supportsResidentMemoryLimitGB: text.contains("--resident-memory-limit-gb"),
            supportsResidentModelIdleTTL: text.contains("--resident-model-idle-ttl")
        )
    }

    static func probe(
        binary: URL,
        timeoutSeconds: TimeInterval = 5
    ) async -> ServerRuntimeCapabilities {
        await Task.detached(priority: .utility) {
            probeBlocking(binary: binary, timeoutSeconds: timeoutSeconds)
        }.value
    }

    private static func probeBlocking(
        binary: URL,
        timeoutSeconds: TimeInterval
    ) -> ServerRuntimeCapabilities {
        let process = Process()
        process.executableURL = binary
        process.arguments = ["serve", "--help"]

        let output = Pipe()
        process.standardOutput = output
        process.standardError = output

        do {
            try process.run()
        } catch {
            return .conservative
        }
        output.fileHandleForWriting.closeFile()

        // Drain help output while the process is running. Waiting for exit
        // before reading can fill the pipe and block the child indefinitely.
        let outputReader = RuntimeProbeOutputReader(output.fileHandleForReading)
        outputReader.start()

        let finished = DispatchSemaphore(value: 0)
        let processHandle = RuntimeProbeProcess(process)
        DispatchQueue.global(qos: .utility).async {
            processHandle.process.waitUntilExit()
            finished.signal()
        }

        let timeout = DispatchTime.now() + .milliseconds(
            max(1, Int(timeoutSeconds * 1000))
        )
        if finished.wait(timeout: timeout) == .timedOut {
            process.terminate()
            if finished.wait(timeout: .now() + .milliseconds(250)) == .timedOut {
                kill(process.processIdentifier, SIGKILL)
                _ = finished.wait(timeout: .now() + .milliseconds(500))
            }
            _ = outputReader.wait(timeout: .now() + .milliseconds(500))
            return .conservative
        }

        outputReader.wait()
        guard process.terminationReason == .exit,
              process.terminationStatus == 0 else {
            return .conservative
        }
        guard let help = String(data: outputReader.data, encoding: .utf8) else {
            return .conservative
        }
        return parse(serveHelp: help)
    }
}

private final class RuntimeProbeProcess: @unchecked Sendable {
    let process: Process

    init(_ process: Process) {
        self.process = process
    }
}

private final class RuntimeProbeOutputReader: @unchecked Sendable {
    private let handle: FileHandle
    private let finished = DispatchSemaphore(value: 0)
    private let lock = NSLock()
    private var captured = Data()

    init(_ handle: FileHandle) {
        self.handle = handle
    }

    func start() {
        DispatchQueue.global(qos: .utility).async { [self] in
            let bytes = handle.readToEndSafely()
            lock.lock()
            captured = bytes
            lock.unlock()
            finished.signal()
        }
    }

    func wait() {
        finished.wait()
    }

    func wait(timeout: DispatchTime) -> DispatchTimeoutResult {
        finished.wait(timeout: timeout)
    }

    var data: Data {
        lock.lock()
        defer { lock.unlock() }
        return captured
    }
}

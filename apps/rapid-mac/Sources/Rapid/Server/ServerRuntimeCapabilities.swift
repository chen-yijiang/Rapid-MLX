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
        timeoutSeconds: TimeInterval = 5,
        ambientEnvironment: [String: String] = ProcessInfo.processInfo.environment
    ) async -> ServerRuntimeCapabilities {
        await Task.detached(priority: .utility) {
            probeBlocking(
                binary: binary,
                timeoutSeconds: timeoutSeconds,
                ambientEnvironment: ambientEnvironment
            )
        }.value
    }

    private static func probeBlocking(
        binary: URL,
        timeoutSeconds: TimeInterval,
        ambientEnvironment: [String: String]
    ) -> ServerRuntimeCapabilities {
        let deadline = DispatchTime.now() + .milliseconds(
            max(1, Int(timeoutSeconds * 1000))
        )
        let standardOutput = Pipe()
        let standardError = Pipe()
        let finished = DispatchSemaphore(value: 0)
        let child: ProcessGroupChild

        do {
            child = try ProcessGroupChild.spawn(
                executableURL: binary,
                arguments: ["serve", "--help"],
                standardInput: .nullDevice,
                standardOutput: standardOutput,
                standardError: standardError,
                environmentAdditions: ServerManager.serveEnvironmentAdditions(
                    bearer: "",
                    ambient: ambientEnvironment
                ),
                replaceEnvironment: true
            ) { _ in
                finished.signal()
            }
        } catch {
            return .conservative
        }

        // Drain help output while the process is running. Waiting for exit
        // before reading can fill the pipe and block the child indefinitely.
        let stdoutReader = RuntimeProbeOutputReader(standardOutput.fileHandleForReading)
        let stderrReader = RuntimeProbeOutputReader(standardError.fileHandleForReading)
        stdoutReader.start(deadline: deadline)
        stderrReader.start(deadline: deadline)

        if finished.wait(timeout: deadline) == .timedOut {
            terminateProbeGroup(child)
            _ = stdoutReader.wait(timeout: deadline)
            _ = stderrReader.wait(timeout: deadline)
            return .conservative
        }

        guard stdoutReader.wait(timeout: deadline) == .success,
              stderrReader.wait(timeout: deadline) == .success,
              DispatchTime.now() < deadline,
              !child.isProcessGroupAlive else {
            terminateProbeGroup(child)
            return .conservative
        }
        guard child.terminationReason == .exit,
              child.terminationStatus == 0 else {
            return .conservative
        }
        let stdoutCapabilities = stdoutReader.capabilities
        let stderrCapabilities = stderrReader.capabilities
        return ServerRuntimeCapabilities(
            supportsResidentMemoryLimitGB:
                stdoutCapabilities.supportsResidentMemoryLimitGB
                || stderrCapabilities.supportsResidentMemoryLimitGB,
            supportsResidentModelIdleTTL:
                stdoutCapabilities.supportsResidentModelIdleTTL
                || stderrCapabilities.supportsResidentModelIdleTTL
        )
    }

    /// The probe binary is user-selectable, so treat its entire descendant
    /// tree as one bounded child. A helper must not survive a timed-out or
    /// malformed `serve --help` invocation.
    private static func terminateProbeGroup(_ child: ProcessGroupChild) {
        child.signalProcessGroup(SIGTERM)
        let termDeadline = Date().addingTimeInterval(0.25)
        while child.isProcessGroupAlive, Date() < termDeadline {
            Thread.sleep(forTimeInterval: 0.01)
        }
        guard child.isProcessGroupAlive else { return }
        child.signalProcessGroup(SIGKILL)
        let killDeadline = Date().addingTimeInterval(0.5)
        while child.isProcessGroupAlive, Date() < killDeadline {
            Thread.sleep(forTimeInterval: 0.01)
        }
    }
}

private final class RuntimeProbeOutputReader: @unchecked Sendable {
    private static let memoryLimitMarker = Data("--resident-memory-limit-gb".utf8)
    private static let idleTTLMarker = Data("--resident-model-idle-ttl".utf8)
    private static let maximumMarkerLength = max(memoryLimitMarker.count, idleTTLMarker.count)

    private let handle: FileHandle
    private let fileDescriptor: Int32
    private let finished = DispatchSemaphore(value: 0)
    private let lock = NSLock()
    private var detectedCapabilities = ServerRuntimeCapabilities.conservative

    init(_ handle: FileHandle) {
        self.handle = handle
        self.fileDescriptor = handle.fileDescriptor
    }

    func start(deadline: DispatchTime) {
        DispatchQueue.global(qos: .utility).async { [self] in
            let capabilities = readUntilClosedOrDeadline(deadline)
            lock.lock()
            detectedCapabilities = capabilities
            lock.unlock()
            finished.signal()
        }
    }

    func wait(timeout: DispatchTime) -> DispatchTimeoutResult {
        finished.wait(timeout: timeout)
    }

    var capabilities: ServerRuntimeCapabilities {
        lock.lock()
        defer { lock.unlock() }
        return detectedCapabilities
    }

    private func readUntilClosedOrDeadline(
        _ deadline: DispatchTime
    ) -> ServerRuntimeCapabilities {
        var supportsMemoryLimit = false
        var supportsIdleTTL = false
        // Keep only enough trailing bytes to recognize a marker split across
        // two reads. The pipe is always fully drained, but a noisy runtime can
        // no longer grow Desktop memory in proportion to its output.
        var markerTail = Data()

        while true {
            // Capture the clock once. Re-reading it after the comparison can
            // cross the deadline and underflow this UInt64 subtraction.
            let now = DispatchTime.now().uptimeNanoseconds
            guard now < deadline.uptimeNanoseconds else { break }
            let remainingNanoseconds = deadline.uptimeNanoseconds - now
            if remainingNanoseconds < 1_000_000 {
                break
            }

            var fileDescriptorSet = pollfd(
                fd: fileDescriptor,
                events: Int16(POLLIN),
                revents: 0
            )
            let timeoutMilliseconds = Int32(min(
                remainingNanoseconds / 1_000_000,
                UInt64(Int32.max)
            ))
            var pollResult: Int32
            repeat {
                pollResult = poll(&fileDescriptorSet, 1, timeoutMilliseconds)
            } while pollResult == -1 && errno == EINTR
            if pollResult <= 0 {
                break
            }

            var chunk = [UInt8](repeating: 0, count: 4_096)
            let byteCount = read(fileDescriptor, &chunk, chunk.count)
            if byteCount > 0 {
                if !supportsMemoryLimit || !supportsIdleTTL {
                    let bytes = markerTail + Data(chunk[0..<byteCount])
                    if !supportsMemoryLimit,
                       bytes.range(of: Self.memoryLimitMarker) != nil {
                        supportsMemoryLimit = true
                    }
                    if !supportsIdleTTL,
                       bytes.range(of: Self.idleTTLMarker) != nil {
                        supportsIdleTTL = true
                    }
                    let tailCount = min(Self.maximumMarkerLength - 1, bytes.count)
                    markerTail = Data(bytes.suffix(tailCount))
                }
            } else if byteCount == 0 || errno != EINTR {
                break
            }
        }

        return ServerRuntimeCapabilities(
            supportsResidentMemoryLimitGB: supportsMemoryLimit,
            supportsResidentModelIdleTTL: supportsIdleTTL
        )
    }
}

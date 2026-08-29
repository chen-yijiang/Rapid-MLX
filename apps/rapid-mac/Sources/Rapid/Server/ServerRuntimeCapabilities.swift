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
        let deadline = DispatchTime.now() + .milliseconds(
            max(1, Int(timeoutSeconds * 1000))
        )
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
        outputReader.start(deadline: deadline)

        let finished = DispatchSemaphore(value: 0)
        let processHandle = RuntimeProbeProcess(process)
        DispatchQueue.global(qos: .utility).async {
            processHandle.process.waitUntilExit()
            finished.signal()
        }

        if finished.wait(timeout: deadline) == .timedOut {
            process.terminate()
            if finished.wait(timeout: .now() + .milliseconds(250)) == .timedOut {
                kill(process.processIdentifier, SIGKILL)
                _ = finished.wait(timeout: .now() + .milliseconds(500))
            }
            _ = outputReader.wait(timeout: deadline)
            return .conservative
        }

        guard outputReader.wait(timeout: deadline) == .success,
              DispatchTime.now() < deadline else {
            return .conservative
        }
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
    private let fileDescriptor: Int32
    private let finished = DispatchSemaphore(value: 0)
    private let lock = NSLock()
    private var captured = Data()

    init(_ handle: FileHandle) {
        self.handle = handle
        self.fileDescriptor = handle.fileDescriptor
    }

    func start(deadline: DispatchTime) {
        DispatchQueue.global(qos: .utility).async { [self] in
            let bytes = readUntilClosedOrDeadline(deadline)
            lock.lock()
            captured = bytes
            lock.unlock()
            finished.signal()
        }
    }

    func wait(timeout: DispatchTime) -> DispatchTimeoutResult {
        finished.wait(timeout: timeout)
    }

    var data: Data {
        lock.lock()
        defer { lock.unlock() }
        return captured
    }

    private func readUntilClosedOrDeadline(_ deadline: DispatchTime) -> Data {
        var captured = Data()

        while deadline.uptimeNanoseconds > DispatchTime.now().uptimeNanoseconds {
            let remainingNanoseconds = deadline.uptimeNanoseconds
                - DispatchTime.now().uptimeNanoseconds
            if remainingNanoseconds < 1_000_000 {
                break
            }

            var fileDescriptorSet = pollfd(
                fd: fileDescriptor,
                events: Int16(POLLIN),
                revents: 0
            )
            let pollResult = poll(
                &fileDescriptorSet,
                1,
                Int32(min(
                    remainingNanoseconds / 1_000_000,
                    UInt64(Int32.max)
                ))
            )
            if pollResult <= 0 {
                break
            }

            var chunk = [UInt8](repeating: 0, count: 4_096)
            let byteCount = read(fileDescriptor, &chunk, chunk.count)
            if byteCount > 0 {
                captured.append(contentsOf: chunk[0..<byteCount])
            } else if byteCount == 0 || errno != EINTR {
                break
            }
        }

        return Data(captured)
    }
}

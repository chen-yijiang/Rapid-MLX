import Darwin
import Foundation
import Testing
@testable import Rapid

@Suite("Server runtime capability probing")
struct ServerRuntimeCapabilitiesTests {
    @Test("serve help with resident flags enables both residency arguments")
    func parseCurrentServeHelp() {
        let capabilities = ServerRuntimeCapabilities.parse(serveHelp: """
        usage: rapid-mlx serve [-h] [--resident-memory-limit-gb RESIDENT_MEMORY_LIMIT_GB]
                               [--resident-model-idle-ttl RESIDENT_MODEL_IDLE_TTL]
        """)

        #expect(capabilities.supportsResidentMemoryLimitGB)
        #expect(capabilities.supportsResidentModelIdleTTL)
    }

    @Test("serve help without resident flags disables both residency arguments")
    func parseOldServeHelp() {
        let capabilities = ServerRuntimeCapabilities.parse(serveHelp: """
        usage: rapid-mlx serve [-h] [--served-model-name SERVED_MODEL_NAME]
        """)

        #expect(!capabilities.supportsResidentMemoryLimitGB)
        #expect(!capabilities.supportsResidentModelIdleTTL)
    }

    @Test("resident launch flags are omitted for older runtimes")
    func oldRuntimeDoesNotReceiveResidentFlags() {
        let flags = ServerManager.residentLaunchFlags(
            memoryCeilingGB: 14,
            capabilities: .conservative
        )

        #expect(flags.isEmpty)
    }

    @Test("resident launch flags are emitted for runtimes that advertise them")
    func currentRuntimeReceivesResidentFlags() {
        let flags = ServerManager.residentLaunchFlags(
            memoryCeilingGB: 14,
            capabilities: .allKnown
        )

        #expect(flags == [
            "--resident-memory-limit-gb", "14",
            "--resident-model-idle-ttl", "1800",
        ])
    }

    @Test("probe reads the selected runtime serve help")
    func probeRuntimeHelp() async throws {
        let runtime = try makeRuntimeScript()

        let capabilities = await ServerRuntimeCapabilities.probe(
            binary: runtime,
            timeoutSeconds: 5
        )

        #expect(capabilities == .allKnown)
    }

    @Test("probe drains help output larger than the pipe buffer")
    func probeLargeRuntimeHelp() async throws {
        let runtime = try makeRuntimeScript(helpPaddingLines: 4_096)

        let capabilities = await ServerRuntimeCapabilities.probe(
            binary: runtime,
            timeoutSeconds: 2
        )

        #expect(capabilities == .allKnown)
    }

    @Test("probe ignores help text from a failed runtime command")
    func probeFailedRuntimeHelp() async throws {
        let runtime = try makeRuntimeScript(helpExitStatus: 7)

        let capabilities = await ServerRuntimeCapabilities.probe(
            binary: runtime,
            timeoutSeconds: 2
        )

        #expect(capabilities == .conservative)
    }

    @Test("probe falls back conservatively when the runtime does not run")
    func probeFailureIsConservative() async {
        let missing = URL(fileURLWithPath: "/tmp/rapid-mlx-missing-\(UUID().uuidString)")

        let capabilities = await ServerRuntimeCapabilities.probe(
            binary: missing,
            timeoutSeconds: 1
        )

        #expect(capabilities == .conservative)
    }

    private func makeRuntimeScript(
        helpPaddingLines: Int = 0,
        helpExitStatus: Int = 0
    ) throws -> URL {
        let directory = FileManager.default.temporaryDirectory
            .appendingPathComponent("rapid-runtime-capabilities-\(UUID().uuidString)")
        try FileManager.default.createDirectory(
            at: directory,
            withIntermediateDirectories: true
        )
        let script = directory.appendingPathComponent("rapid-mlx")
        try """
        #!/bin/sh
        if [ "$1" = "serve" ] && [ "$2" = "--help" ]; then
          i=0
          while [ "$i" -lt \(helpPaddingLines) ]; do
            echo '0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef'
            i=$((i + 1))
          done
          echo '--resident-memory-limit-gb'
          echo '--resident-model-idle-ttl'
          exit \(helpExitStatus)
        fi
        exit 2
        """.write(to: script, atomically: true, encoding: .utf8)
        chmod(script.path, 0o755)
        return script
    }
}

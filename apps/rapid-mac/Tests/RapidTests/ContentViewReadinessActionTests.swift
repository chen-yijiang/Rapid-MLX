import Foundation
import Testing
@testable import Rapid

@Suite("ContentView readiness action wiring")
struct ContentViewReadinessActionTests {
    private static var packageRoot: URL {
        URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent()  // RapidTests
            .deletingLastPathComponent()  // Tests
            .deletingLastPathComponent()  // package root
    }

    @Test("Chat readiness replaces a different resident model")
    func readinessUsesEnsureServing() throws {
        let url = Self.packageRoot
            .appendingPathComponent("Sources/Rapid/UI/ContentView.swift")
        let body = try String(contentsOf: url, encoding: .utf8)
        let source = CapabilityChipRenderGateSourceGuardTests
            .stripCommentsAndWhitespace(body)

        guard let signature = source.range(
            of: "privatefuncstartModel(_target:String){"
        ) else {
            Issue.record("ContentView.startModel could not be found")
            return
        }
        guard let function = Self.balancedBody(
            of: source,
            openingBraceAt: source.index(before: signature.upperBound)
        ) else {
            Issue.record("ContentView.startModel has no closing brace")
            return
        }

        #expect(
            function.contains("server.ensureServing(alias:target,hfPath:hfPath)"),
            "The Chat readiness button must switch away from a resident Images model."
        )
        #expect(
            !function.contains("server.start(alias:target,hfPath:hfPath)"),
            "ServerManager.start is cold-start only and silently no-ops while Images is resident."
        )
    }

    /// The source text of the block whose opening `{` is at `start`, up to
    /// and including its matching `}`.
    ///
    /// Slicing to the FIRST `}` instead does not work here, and that is not a
    /// stylistic point: `startModel` opens a closure — `first(where: { … })` —
    /// before it reaches the call this suite is about, so the first closing
    /// brace belongs to that closure. A first-brace slice therefore ends at
    /// `…first(where:{$0.alias==target}` and cannot contain `ensureServing`
    /// no matter how the function is written, which leaves the gate
    /// permanently red instead of protecting anything.
    ///
    /// Brace counting is fooled by a `{` or `}` inside a string literal.
    /// `startModel` has none, and if one ever appears the count goes wrong in
    /// the direction that fails this test loudly rather than passing it
    /// silently.
    private static func balancedBody(
        of source: String,
        openingBraceAt start: String.Index
    ) -> String? {
        var depth = 0
        var index = start
        while index < source.endIndex {
            switch source[index] {
            case "{":
                depth += 1
            case "}":
                depth -= 1
                if depth == 0 { return String(source[start...index]) }
            default:
                break
            }
            index = source.index(after: index)
        }
        return nil
    }
}

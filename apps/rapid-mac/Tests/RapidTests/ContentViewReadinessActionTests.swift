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

        // Brace counting is fooled by a `{` or `}` inside a string literal, so
        // the slice can end early (or, if a literal swallows a real brace,
        // late). Neither is allowed to hide a regression, so each check below
        // is arranged to fail rather than pass when the slice is wrong.

        // A slice that ended early cannot contain the call, so this goes red —
        // loudly, which is the safe direction.
        #expect(
            function.contains("server.ensureServing(alias:target,hfPath:hfPath)"),
            "The Chat readiness button must switch away from a resident Images model."
        )

        // Deliberately checked against the WHOLE file, not the slice. Scoping
        // it to the slice is what makes a mis-slice dangerous: put
        // `let marker = "}"` after the ensureServing call and before a
        // cold-start call, and the extraction stops at the literal's brace —
        // the positive check above still passes, and the forbidden call sits
        // just outside the slice, unseen. Against the whole file there is
        // nowhere outside the slice to hide.
        //
        // ContentView does call `server.start` elsewhere (resume, confirmed,
        // newAlias), but never with this argument shape, so the exact stripped
        // form below is specific to startModel's signature.
        #expect(
            !source.contains("server.start(alias:target,hfPath:hfPath)"),
            "ServerManager.start is cold-start only and silently no-ops while Images is resident."
        )

        // The remaining risk is a slice that ran LONG — then the positive check
        // could be satisfied by some other function's call. A second
        // declaration inside the body is the cheap tell.
        #expect(
            !function.dropFirst().contains("privatefunc"),
            "the extracted body ran past startModel into the next declaration, so the check above is no longer about startModel"
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
    /// This counts braces without lexing string literals. That is deliberate:
    /// a half-correct Swift lexer in a test is its own hazard, and the caller
    /// is arranged so that a wrong slice fails the suite instead of passing
    /// it. See the comments at each expectation.
    ///
    /// The cost of that choice is a false failure, not a false pass: add a
    /// literal `"{"` or `"}"` to `startModel` and this suite goes red on
    /// correct code. That is the point at which to teach this helper about
    /// string literals — the test will have told you.
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

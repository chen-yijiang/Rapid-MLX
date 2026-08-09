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
        let source = Self.canonicalSource(try String(contentsOf: url, encoding: .utf8))

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

        // Checked against the whole file rather than the slice. ContentView
        // does call `server.start` elsewhere — resume, confirmed, newAlias —
        // but never with this argument shape, so the form below is specific to
        // startModel's signature, and checking it file-wide means a slice that
        // ends early cannot leave a forbidden call sitting just outside it.
        #expect(
            !source.contains("server.start(alias:target,hfPath:hfPath)"),
            "ServerManager.start is cold-start only and silently no-ops while Images is resident."
        )

        // Tripwire, not a proof: if the extraction ever runs past startModel,
        // the check above could be satisfied by a different function's call.
        // These are the two declaration forms that follow it today.
        for declaration in ["privatefunc", "privatevar"] {
            #expect(
                !function.dropFirst().contains(declaration),
                "the extracted body ran past startModel into a following \(declaration) declaration"
            )
        }
    }

    // MARK: - Source canonicalisation

    /// `swift` with comments and whitespace removed, and every string literal
    /// reduced to an empty one.
    ///
    /// Erasing literal *contents* is what makes the plain text matching in
    /// this suite sound. Two bypasses fall out of it, both of which were live
    /// before this existed:
    ///
    /// - a `"}"` literal ends the brace scan early, so a forbidden call placed
    ///   after it sits outside the extracted body;
    /// - a `"//"` literal makes a comment stripper eat the rest of the line,
    ///   deleting real executable code from the text being asserted on.
    ///
    /// Both need the scanner to know where literals begin and end. Once it
    /// does, the cheapest thing to do with the contents is drop them: no
    /// literal can then contribute a brace, a slash, or a call.
    ///
    /// Not handled: a call written inside a string interpolation, whose
    /// contents are erased along with the rest of the literal. That is not a
    /// shape this gate is about, and it fails safe for the positive check.
    static func canonicalSource(_ swift: String) -> String {
        var out = ""
        out.reserveCapacity(swift.count)
        var i = swift.startIndex

        func peek(_ offset: Int, from index: String.Index) -> Character? {
            guard let j = swift.index(index, offsetBy: offset, limitedBy: swift.endIndex),
                j < swift.endIndex
            else { return nil }
            return swift[j]
        }

        while i < swift.endIndex {
            let c = swift[i]

            // Line comment.
            if c == "/", peek(1, from: i) == "/" {
                while i < swift.endIndex, swift[i] != "\n" { i = swift.index(after: i) }
                continue
            }
            // Block comment. Swift nests them.
            if c == "/", peek(1, from: i) == "*" {
                var depth = 0
                while i < swift.endIndex {
                    if swift[i] == "/", peek(1, from: i) == "*" {
                        depth += 1
                        i = swift.index(i, offsetBy: 2)
                    } else if swift[i] == "*", peek(1, from: i) == "/" {
                        depth -= 1
                        i = swift.index(i, offsetBy: 2)
                        if depth == 0 { break }
                    } else {
                        i = swift.index(after: i)
                    }
                }
                continue
            }
            // String literal, including raw (`#"…"#`) and multiline (`"""`).
            if c == "\"" || c == "#" {
                if let end = Self.endOfStringLiteral(in: swift, at: i) {
                    out += "\"\""
                    i = end
                    continue
                }
                // A `#` that does not open a raw literal (`#available`,
                // `#filePath`) is ordinary source.
            }
            if !c.isWhitespace { out.append(c) }
            i = swift.index(after: i)
        }
        return out
    }

    /// The index just past the closing delimiter of the string literal
    /// starting at `start`, or nil if no literal starts there.
    private static func endOfStringLiteral(
        in source: String,
        at start: String.Index
    ) -> String.Index? {
        var cursor = start
        var hashes = 0
        while cursor < source.endIndex, source[cursor] == "#" {
            hashes += 1
            cursor = source.index(after: cursor)
        }
        guard cursor < source.endIndex, source[cursor] == "\"" else { return nil }

        // A run of three or more opening quotes is a multiline literal; `""`
        // on its own is just an empty one.
        var quoteRun = 0
        var probe = cursor
        while probe < source.endIndex, source[probe] == "\"" {
            quoteRun += 1
            probe = source.index(after: probe)
        }
        let delimiter = quoteRun >= 3 ? 3 : 1

        var scan = source.index(cursor, offsetBy: delimiter)
        while scan < source.endIndex {
            // An escape is `\` in a normal literal and `\###…` in a raw one
            // with that many hashes; anything else is a literal backslash.
            if source[scan] == "\\" {
                var probe = source.index(after: scan)
                var seen = 0
                while seen < hashes, probe < source.endIndex, source[probe] == "#" {
                    seen += 1
                    probe = source.index(after: probe)
                }
                if seen == hashes {
                    scan = probe < source.endIndex ? source.index(after: probe) : probe
                    continue
                }
            }
            if source[scan] == "\"" {
                // `delimiter` quotes followed by `hashes` hashes closes it.
                var probe = scan
                var quotes = 0
                while quotes < delimiter, probe < source.endIndex, source[probe] == "\"" {
                    quotes += 1
                    probe = source.index(after: probe)
                }
                if quotes == delimiter {
                    var seen = 0
                    while seen < hashes, probe < source.endIndex, source[probe] == "#" {
                        seen += 1
                        probe = source.index(after: probe)
                    }
                    if seen == hashes { return probe }
                }
                scan = probe
                continue
            }
            scan = source.index(after: scan)
        }
        return nil  // Unterminated: treat as "not a literal" and keep scanning.
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
    /// Counting braces is only sound because ``canonicalSource(_:)`` has
    /// already emptied every string literal, so no brace here is inside one.
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

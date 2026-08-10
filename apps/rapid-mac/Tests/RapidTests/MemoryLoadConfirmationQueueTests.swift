import Foundation
import Testing
@testable import Rapid

@Suite("Memory-load confirmation request isolation (#1463)")
struct MemoryLoadConfirmationQueueTests {
    private func warning(_ alias: String) -> ModelSizing.MemoryWarning {
        ModelSizing.MemoryWarning(
            alias: alias,
            hfPath: nil,
            isAutoRespawn: false,
            severity: .unsafe,
            footprintGB: 24,
            freeGB: 4,
            totalGB: 32
        )
    }

    @Test("overlapping loads receive only their own decisions")
    func decisionsAreRequestScoped() {
        var queue = MemoryLoadConfirmationQueue()
        let requestA = UUID()
        let requestB = UUID()
        let warningA = warning("model-a")
        let warningB = warning("model-b")

        queue.enqueue(warning: warningA, requestID: requestA)
        queue.enqueue(warning: warningB, requestID: requestB)

        #expect(queue.currentWarning?.id == warningA.id)
        #expect(queue.resolveCurrent(warning: warningB, decision: .confirmed(sequence: 99)) == false)
        #expect(queue.resolveCurrent(warning: warningA, decision: .cancelled) == true)
        #expect(queue.takeDecision(for: requestA) == .cancelled)
        #expect(queue.takeDecision(for: requestB) == nil)

        #expect(queue.currentWarning?.id == warningB.id)
        #expect(queue.resolveCurrent(warning: warningB, decision: .confirmed(sequence: 7)) == true)
        #expect(queue.takeDecision(for: requestB) == .confirmed(sequence: 7))
        #expect(queue.currentWarning == nil)
    }

    @Test("same-alias loads remain distinct requests")
    func duplicateAliasesRemainDistinct() {
        var queue = MemoryLoadConfirmationQueue()
        let requestA = UUID()
        let requestB = UUID()
        let warningA = warning("same-model")
        let warningB = warning("same-model")

        queue.enqueue(warning: warningA, requestID: requestA)
        queue.enqueue(warning: warningB, requestID: requestB)
        #expect(warningA.id != warningB.id)

        #expect(queue.resolveCurrent(warning: warningA, decision: .confirmed(sequence: 1)))
        #expect(queue.takeDecision(for: requestA) == .confirmed(sequence: 1))
        #expect(queue.takeDecision(for: requestB) == nil)
        #expect(queue.currentWarning?.id == warningB.id)
    }

    @Test("direct starts do not retain an unconsumed decision")
    func directStartDecisionIsNotRetained() {
        var queue = MemoryLoadConfirmationQueue()
        let warning = warning("picker-start")
        queue.enqueue(warning: warning, requestID: nil)

        #expect(queue.resolveCurrent(warning: warning, decision: .confirmed(sequence: 3)))
        #expect(queue.currentWarning == nil)
    }

    @Test("a cancelled waiter leaves the visible prompt usable without leaking a result")
    func abandonedWaiterDoesNotLeakDecision() {
        var queue = MemoryLoadConfirmationQueue()
        let request = UUID()
        let warning = warning("cancelled-chat")
        queue.enqueue(warning: warning, requestID: request)

        queue.abandonWaiter(request)
        #expect(queue.currentWarning?.id == warning.id)
        #expect(queue.resolveCurrent(warning: warning, decision: .confirmed(sequence: 4)))
        #expect(queue.takeDecision(for: request) == nil)
    }
}

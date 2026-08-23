import Foundation
import Testing

@testable import Rapid

@Suite("Chat attachment draft state")
struct ChatAttachmentDraftTests {
    @Test("consume atomically clears attachments, identity, notice, and import state")
    func consumeClearsEveryTransientField() throws {
        let image = try makeImage(name: "first.png")
        let file = try makeFile(name: "notes.txt", text: "first turn")
        let imageURL = URL(fileURLWithPath: "/tmp/first.png")
        let fileURL = URL(fileURLWithPath: "/tmp/notes.txt")
        var draft = ChatAttachmentDraft()
        draft.appendImage(image, sourceURL: imageURL)
        let startedImportID = draft.beginFileImport(conversationID: UUID())
        let importID = try #require(startedImportID)
        draft.finishFileImport(id: importID, [(file, fileURL)], notice: "old notice")
        _ = draft.beginFileImport(conversationID: UUID())

        let payload = draft.takeSubmission()

        #expect(payload.images == [image])
        #expect(payload.files == [file])
        #expect(!draft.hasAttachments)
        #expect(draft.sourcePaths.isEmpty)
        #expect(draft.notice == nil)
        #expect(!draft.isImportingFiles)
    }

    @Test("a second turn cannot inherit the first turn's image or file")
    func sequentialTurnsRemainIsolated() throws {
        let firstImage = try makeImage(name: "first.png")
        let secondImage = try makeImage(name: "second.png")
        let firstFile = try makeFile(name: "first.txt", text: "alpha")
        let secondFile = try makeFile(name: "second.txt", text: "beta")
        var draft = ChatAttachmentDraft()

        draft.appendImage(firstImage)
        let startedFirstImportID = draft.beginFileImport(conversationID: UUID())
        let firstImportID = try #require(startedFirstImportID)
        draft.finishFileImport(
            id: firstImportID,
            [(firstFile, URL(fileURLWithPath: "/tmp/first.txt"))],
            notice: nil
        )
        let first = draft.takeSubmission()

        draft.appendImage(secondImage)
        let startedSecondImportID = draft.beginFileImport(conversationID: UUID())
        let secondImportID = try #require(startedSecondImportID)
        draft.finishFileImport(
            id: secondImportID,
            [(secondFile, URL(fileURLWithPath: "/tmp/second.txt"))],
            notice: nil
        )
        let second = draft.takeSubmission()

        #expect(first.images.map(\.filename) == ["first.png"])
        #expect(first.files.map(\.filename) == ["first.txt"])
        #expect(second.images.map(\.filename) == ["second.png"])
        #expect(second.files.map(\.filename) == ["second.txt"])
    }

    @Test("removing an attachment also releases its source identity")
    func removalReleasesIdentity() throws {
        let url = URL(fileURLWithPath: "/tmp/reusable.png")
        let image = try makeImage(name: "reusable.png")
        var draft = ChatAttachmentDraft()
        draft.appendImage(image, sourceURL: url)
        #expect(draft.filteringAlreadyAttached([url]).duplicates == 1)

        draft.removeImage(id: image.id)

        #expect(draft.filteringAlreadyAttached([url]).fresh == [url])
    }

    @Test("all input methods share one path-normalized duplicate filter")
    func duplicateFilterCoversExistingAndBatchDuplicates() {
        let existing = URL(fileURLWithPath: "/tmp/docs/photo.png")
        let sameSpelling = URL(fileURLWithPath: "/tmp/docs/../docs/photo.png")
        let fresh = URL(fileURLWithPath: "/tmp/new.txt")
        var draft = ChatAttachmentDraft()
        let image = try? makeImage(name: "photo.png")
        if let image { draft.appendImage(image, sourceURL: existing) }

        let result = draft.filteringAlreadyAttached([sameSpelling, fresh, fresh])

        #expect(result.fresh == [fresh])
        #expect(result.duplicates == 2)
    }

    @Test("a cancelled import cannot resurrect attachments when it completes late")
    func staleImportCompletionIsIgnored() throws {
        let staleFile = try makeFile(name: "old.txt", text: "old conversation")
        let staleURL = URL(fileURLWithPath: "/tmp/old.txt")
        var draft = ChatAttachmentDraft()
        let startedStaleID = draft.beginFileImport(conversationID: UUID())
        let staleID = try #require(startedStaleID)

        let cancelled = draft.cancelFileImport(notice: "Import canceled after navigation.")
        let accepted = draft.finishFileImport(
            id: staleID,
            [(staleFile, staleURL)],
            notice: "stale notice"
        )

        #expect(!accepted)
        #expect(draft.files.isEmpty)
        #expect(cancelled)
        #expect(draft.notice == "Import canceled after navigation.")
        #expect(!draft.isImportingFiles)
        #expect(draft.filteringAlreadyAttached([staleURL]).fresh == [staleURL])
    }

    @Test("a late old generation cannot overwrite a newer import")
    func importGenerationsDoNotCross() throws {
        let oldFile = try makeFile(name: "old.txt", text: "old")
        let newFile = try makeFile(name: "new.txt", text: "new")
        var draft = ChatAttachmentDraft()
        let startedOldID = draft.beginFileImport(conversationID: UUID())
        let oldID = try #require(startedOldID)
        let cancelledOld = draft.cancelFileImport()
        #expect(cancelledOld)
        let startedNewID = draft.beginFileImport(conversationID: UUID())
        let newID = try #require(startedNewID)

        let acceptedOld = draft.finishFileImport(
            id: oldID,
            [(oldFile, URL(fileURLWithPath: "/tmp/old.txt"))],
            notice: nil
        )
        #expect(!acceptedOld)
        #expect(draft.isImportingFiles)
        let acceptedNew = draft.finishFileImport(
            id: newID,
            [(newFile, URL(fileURLWithPath: "/tmp/new.txt"))],
            notice: nil
        )
        #expect(acceptedNew)
        #expect(draft.files == [newFile])
    }

    @Test("a stale generation cannot cancel the newer active import")
    func staleGenerationCannotCancelNewImport() throws {
        var draft = ChatAttachmentDraft()
        let startedOldID = draft.beginFileImport(conversationID: UUID())
        let oldID = try #require(startedOldID)
        _ = draft.cancelFileImport(id: oldID)
        let startedNewID = draft.beginFileImport(conversationID: UUID())
        let newID = try #require(startedNewID)

        let cancelled = draft.cancelFileImport(
            id: oldID,
            notice: "Old conversation changed."
        )

        #expect(!cancelled)
        #expect(draft.fileImport?.id == newID)
        #expect(draft.notice == nil)
    }

    @Test("delayed navigation callback preserves the new conversation's import")
    func delayedNavigationPreservesNewConversationImport() throws {
        let conversationA = UUID()
        let conversationB = UUID()
        var draft = ChatAttachmentDraft()
        let startedA = draft.beginFileImport(conversationID: conversationA)
        let importA = try #require(startedA)
        _ = draft.cancelFileImport(id: importA)
        let startedB = draft.beginFileImport(conversationID: conversationB)
        let importB = try #require(startedB)

        // Models SwiftUI delivering A -> B after B has already begun work.
        let cancelled = draft.cancelFileImport(
            notOwnedBy: conversationB,
            notice: "Import canceled after navigation."
        )

        #expect(!cancelled)
        #expect(draft.fileImport == .init(id: importB, conversationID: conversationB))
        #expect(draft.notice == nil)
    }

    @Test("submission is an immutable snapshot of one composer turn")
    func submissionDoesNotFollowLaterDraftMutations() throws {
        let first = try makeImage(name: "first.png")
        let second = try makeImage(name: "second.png")
        var draft = ChatAttachmentDraft()
        draft.appendImage(first)

        let submission = draft.takeSubmission()
        draft.appendImage(second)

        #expect(submission.images == [first])
        #expect(draft.images == [second])
    }

    @Test("ChatView carries the import generation through async work and cancels on navigation")
    func lifecycleIsWiredIntoChatView() throws {
        let source = try String(
            contentsOf: URL(fileURLWithPath: #filePath)
                .deletingLastPathComponent()
                .deletingLastPathComponent()
                .deletingLastPathComponent()
                .appendingPathComponent("Sources/Rapid/UI/ChatView.swift"),
            encoding: .utf8
        )
        let stripped = CapabilityChipRenderGateSourceGuardTests
            .stripCommentsAndWhitespace(source)

        #expect(stripped.contains("letimportConversationID=viewModel.activeConversationID"))
        #expect(stripped.contains(
            "letimportID=attachmentDraft.beginFileImport(conversationID:importConversationID)"
        ))
        #expect(stripped.contains(
            "guardviewModel.activeConversationID==importConversationIDelse{cancelFileImportAfterNavigation(importID:importID)"
        ))
        #expect(stripped.contains("attachmentDraft.finishFileImport(id:importID"))
        #expect(stripped.contains(
            ".onChange(of:viewModel.activeConversationID){_,newConversationIDincancelFileImportAfterNavigation(activeConversationID:newConversationID)"
        ))
    }

    private func makeImage(name: String) throws -> ChatImageAttachment {
        try ChatImageAttachment(
            filename: name,
            mimeType: "image/png",
            data: Data([0x89, 0x50, 0x4E, 0x47])
        )
    }

    private func makeFile(name: String, text: String) throws -> ChatFileAttachment {
        try ChatFileAttachment(
            filename: name,
            kind: .txt,
            extractedText: text,
            sourceByteCount: text.utf8.count
        )
    }
}

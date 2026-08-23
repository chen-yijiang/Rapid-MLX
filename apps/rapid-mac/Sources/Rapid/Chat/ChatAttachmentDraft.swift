import Foundation

/// Product state for attachments waiting in the Chat composer.
///
/// Keeping this outside ``ChatView`` makes the important identity and lifecycle
/// rules directly testable: every input method feeds the same draft, a send
/// atomically consumes it, and a later turn cannot inherit stale attachments.
struct ChatAttachmentDraft: Equatable {
    struct FileImport: Equatable {
        let id: UUID
        let conversationID: UUID
    }

    /// Immutable ownership transfer from the composer to one user turn.
    ///
    /// The arrays are captured before asynchronous chat work starts, so later
    /// composer mutations cannot alter an in-flight request.
    struct Submission: Equatable {
        let images: [ChatImageAttachment]
        let files: [ChatFileAttachment]
    }

    private(set) var images: [ChatImageAttachment] = []
    private(set) var files: [ChatFileAttachment] = []
    private(set) var sourcePaths: [UUID: String] = [:]
    private(set) var fileImport: FileImport?
    var notice: String?

    var hasAttachments: Bool { !images.isEmpty || !files.isEmpty }
    var isImportingFiles: Bool { fileImport != nil }

    mutating func appendImage(_ image: ChatImageAttachment, sourceURL: URL? = nil) {
        images.append(image)
        if let sourceURL { sourcePaths[image.id] = Self.attachmentKey(for: sourceURL) }
    }

    mutating func appendImages(
        _ imported: [(attachment: ChatImageAttachment, sourceURL: URL)]
    ) {
        for item in imported { appendImage(item.attachment, sourceURL: item.sourceURL) }
    }

    /// Starts one asynchronous import generation. A second source cannot race
    /// the first because every UI entry point funnels through this method.
    mutating func beginFileImport(conversationID: UUID) -> UUID? {
        guard fileImport == nil else { return nil }
        let id = UUID()
        fileImport = FileImport(id: id, conversationID: conversationID)
        return id
    }

    /// Applies results only to the generation that created them.
    ///
    /// A conversation transition can cancel an import while file parsing is
    /// still off-main-thread. Its late completion must not resurrect stale
    /// attachments in the current composer.
    @discardableResult
    mutating func finishFileImport(
        id: UUID,
        _ imported: [(attachment: ChatFileAttachment, sourceURL: URL)],
        notice: String?
    ) -> Bool {
        guard fileImport?.id == id else { return false }
        files = ChatFileAttachment.fittedForMessage(files + imported.map(\.attachment))
        for item in imported {
            sourcePaths[item.attachment.id] = Self.attachmentKey(for: item.sourceURL)
        }
        self.notice = notice
        fileImport = nil
        return true
    }

    /// Invalidates the active generation and leaves a visible explanation.
    /// Returns whether an import was actually cancelled so callers can avoid
    /// announcing navigation that had no attachment work in flight.
    @discardableResult
    mutating func cancelFileImport(id expectedID: UUID? = nil, notice: String? = nil) -> Bool {
        guard let activeImport = fileImport else { return false }
        guard expectedID == nil || expectedID == activeImport.id else { return false }
        fileImport = nil
        if let notice { self.notice = notice }
        return true
    }

    /// Cancels only work owned by a conversation other than the active one.
    /// This remains correct even when SwiftUI delivers navigation after the new
    /// conversation has already started its own import.
    @discardableResult
    mutating func cancelFileImport(
        notOwnedBy conversationID: UUID,
        notice: String? = nil
    ) -> Bool {
        guard let activeImport = fileImport else { return false }
        guard activeImport.conversationID != conversationID else { return false }
        return cancelFileImport(id: activeImport.id, notice: notice)
    }

    mutating func removeImage(id: UUID) {
        images.removeAll { $0.id == id }
        sourcePaths[id] = nil
    }

    mutating func removeFile(id: UUID) {
        files.removeAll { $0.id == id }
        sourcePaths[id] = nil
    }

    /// Returns exactly one turn's attachments and clears all transient state.
    /// This is intentionally one mutation so a new import cannot observe an
    /// old source-path map after the visible chips have already disappeared.
    mutating func takeSubmission() -> Submission {
        let submission = Submission(images: images, files: files)
        images = []
        files = []
        sourcePaths = [:]
        notice = nil
        fileImport = nil
        return submission
    }

    func filteringAlreadyAttached(_ urls: [URL]) -> (fresh: [URL], duplicates: Int) {
        Self.withoutAlreadyAttached(urls, attached: Set(sourcePaths.values))
    }

    /// Identity for "the same file". Symlinks and `..` are resolved; equal
    /// bytes at distinct real paths deliberately remain separate attachments.
    static func attachmentKey(for url: URL) -> String {
        url.standardizedFileURL.resolvingSymlinksInPath().path
    }

    static func withoutAlreadyAttached(
        _ urls: [URL], attached: Set<String>
    ) -> (fresh: [URL], duplicates: Int) {
        var seen = attached
        var fresh: [URL] = []
        for url in urls where seen.insert(attachmentKey(for: url)).inserted {
            fresh.append(url)
        }
        return (fresh, urls.count - fresh.count)
    }
}

import Foundation
import Observation

/// State + orchestration for the Images tab. Mirrors ``ChatViewModel``:
/// an ``@Observable`` store the view binds to, owning the image client and
/// the results, and reading ``ServerManager.activePort`` / ``activeBearer``
/// at request time (never caching — they change across a reload).
@MainActor
@Observable
final class ImageGenViewModel {
    /// The prompt the user is composing.
    var prompt: String = ""
    /// The selected image-gen alias (e.g. ``flux-schnell-4bit``).
    var selectedAlias: String = ""
    /// Output size, ``WIDTHxHEIGHT``. Kept to a small curated set in the UI.
    var size: String = "1024x1024"
    /// Newest-first gallery of results this session.
    var results: [GeneratedImage] = []
    /// True while a generate/edit request is in flight — gates the button.
    var isGenerating: Bool = false
    /// User-facing error from the last request, or nil.
    var errorMessage: String?
    /// Image-gen aliases discovered from the catalog (``[image:gen]`` rows).
    var imageModels: [ModelEntry] = []
    /// False until the first catalog refresh completes.
    var catalogLoaded: Bool = false
    /// When set, the next run is an EDIT of this image rather than a fresh
    /// generation — the compose box shows an "editing" affordance.
    var editSource: GeneratedImage?

    private let client = ImageClient()
    private let server: ServerManager

    init(server: ServerManager) {
        self.server = server
    }

    /// True when the selected alias is an instruction-edit model (its alias
    /// carries ``image-edit``). Editing requires such a model to be loaded.
    var selectedIsEditModel: Bool {
        selectedAlias.localizedCaseInsensitiveContains("image-edit")
    }

    /// Whether Generate/Edit can fire: a prompt, a model, and no request in
    /// flight. (Editing additionally needs an ``editSource``.)
    var canSubmit: Bool {
        !isGenerating
            && !prompt.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
            && !selectedAlias.isEmpty
    }

    /// Load the image-gen alias catalog (safe to call repeatedly).
    func refreshCatalog() async {
        guard let binary = server.binaryPath else { return }
        let entries = await ModelCatalog.imageEntries(binary: binary)
        imageModels = entries
        catalogLoaded = true
        if selectedAlias.isEmpty {
            // Prefer a cached model so the first run doesn't force a pull.
            selectedAlias = (entries.first { $0.cached } ?? entries.first)?.alias ?? ""
        }
    }

    /// Generate (or edit, when ``editSource`` is set) from the current prompt.
    func submit() async {
        if let source = editSource {
            await runEdit(source: source)
        } else {
            await runGenerate()
        }
    }

    private func runGenerate() async {
        let trimmed = prompt.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty, !selectedAlias.isEmpty else { return }
        await withRequest {
            let hf = self.imageModels.first { $0.alias == self.selectedAlias }?.hfRepo
            guard await self.server.ensureServing(alias: self.selectedAlias, hfPath: hf) else {
                throw ImageClientError.notReady
            }
            let images = try await self.client.generate(
                prompt: trimmed,
                model: self.selectedAlias,
                size: self.size,
                count: 1,
                seed: nil,
                port: self.server.activePort,
                bearer: self.server.activeBearer
            )
            self.results.insert(contentsOf: images, at: 0)
        }
    }

    private func runEdit(source: GeneratedImage) async {
        let trimmed = prompt.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty, !selectedAlias.isEmpty else { return }
        await withRequest {
            let hf = self.imageModels.first { $0.alias == self.selectedAlias }?.hfRepo
            guard await self.server.ensureServing(alias: self.selectedAlias, hfPath: hf) else {
                throw ImageClientError.notReady
            }
            let images = try await self.client.edit(
                imagePNG: source.pngData,
                prompt: trimmed,
                model: self.selectedAlias,
                size: self.size,
                count: 1,
                seed: nil,
                port: self.server.activePort,
                bearer: self.server.activeBearer
            )
            self.results.insert(contentsOf: images, at: 0)
        }
    }

    /// Begin editing a specific result: stage it as the edit source and clear
    /// the prompt for the instruction.
    func beginEdit(_ image: GeneratedImage) {
        editSource = image
        prompt = ""
        errorMessage = nil
    }

    /// Cancel an in-progress edit intent, returning to fresh generation.
    func cancelEdit() {
        editSource = nil
    }

    /// Shared request wrapper: flips ``isGenerating``, clears ``editSource``
    /// on success, and funnels every failure into ``errorMessage``.
    private func withRequest(_ body: @escaping () async throws -> Void) async {
        isGenerating = true
        errorMessage = nil
        defer { isGenerating = false }
        do {
            try await body()
            editSource = nil
            prompt = ""
        } catch let error as ImageClientError {
            errorMessage = error.errorDescription
        } catch {
            errorMessage = error.localizedDescription
        }
    }
}

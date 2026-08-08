import Foundation
import Testing
@testable import Rapid

/// Coverage for the image-gen catalog split: `[image:gen]` rows must be
/// parsed for the Images tab AND excluded from the chat catalog, so an image
/// checkpoint can never surface in the chat picker (catalog-integrity, #1603).
@Suite("Image catalog")
struct ImageCatalogTests {
    /// A faithful slice of `rapid-mlx models` output: a chat row, the video
    /// section, and the image section the CLI now emits.
    static let sample = """
      Available models (2 aliases)
      ────────────────────────────
      Alias                 Size       Tools      HF id
      ────────────────────────────
      qwen3.6-27b-4bit      15.0 GiB   hermes     mlx-community/Qwen3.6-27B-4bit
      bonsai-1.7b-2bit      0.9 GiB    —          prism-ml/Bonsai

      Video models (1 aliases)
      ────────────────────────────
      Alias                 Size       Kind        HF id
      ────────────────────────────
      ltx-2.3-mlx-q4        24.0 GiB   [video:gen] notapalindrome/ltx23-mlx-av-q4

      Image models (2 aliases)
      ────────────────────────────
      Alias                 Size       Kind        HF id
      ────────────────────────────
      flux-schnell-4bit     8.9 GiB    [image:gen] dhairyashil/FLUX.1-schnell-mflux-4bit
      qwen-image-edit-4bit  25.4 GiB   [image:gen] OsaurusAI/Qwen-Image-Edit-mflux-q4
    """

    @Test("parseImageRows extracts only the [image:gen] rows")
    func parsesImageRows() {
        let rows = ModelCatalog.parseImageRows(Self.sample)
        #expect(rows.count == 2)

        let aliases = rows.map(\.alias)
        #expect(aliases.contains("flux-schnell-4bit"))
        #expect(aliases.contains("qwen-image-edit-4bit"))
        // No chat / video alias leaks in.
        #expect(!aliases.contains("qwen3.6-27b-4bit"))
        #expect(!aliases.contains("ltx-2.3-mlx-q4"))

        let flux = rows.first { $0.alias == "flux-schnell-4bit" }
        #expect(flux?.hfRepo == "dhairyashil/FLUX.1-schnell-mflux-4bit")
        #expect(flux?.size == "8.9 GiB")
    }

    @Test("[image:gen] rows are excluded from the chat catalog")
    func imageRowsExcludedFromChat() {
        // hasNonChatKindTag now drops image alongside audio/video.
        #expect(ModelCatalog.hasNonChatKindTag(
            "flux-schnell-4bit  8.9 GiB  [image:gen] dhairyashil/FLUX.1-schnell-mflux-4bit"))
        #expect(ModelCatalog.hasNonChatKindTag(
            "ltx-2.3-mlx-q4  24.0 GiB  [video:gen] repo/ltx"))
        // A plain chat row is not dropped.
        #expect(!ModelCatalog.hasNonChatKindTag(
            "qwen3.6-27b-4bit  15.0 GiB  hermes  mlx-community/Qwen3.6-27B-4bit"))

        // The chat parser drops the image alias entirely.
        let excluded = ModelCatalog.parseExcludedAliases(Self.sample)
        #expect(excluded.contains("flux-schnell-4bit"))
        #expect(excluded.contains("qwen-image-edit-4bit"))
    }
}

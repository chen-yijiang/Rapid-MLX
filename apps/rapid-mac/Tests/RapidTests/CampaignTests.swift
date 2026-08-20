import Testing
@testable import Rapid

@Suite("Campaign MVP")
struct CampaignTests {
    @Test("preview is opt-in and carries a typed pull action")
    func previewFixture() {
        #expect(Campaign.previewFromEnvironment([:]) == nil)
        let campaign = Campaign.previewFromEnvironment(["RAPID_GUI_CAMPAIGN_PREVIEW": "1"])
        #expect(campaign?.id == "model-qwen35-35b-202608")
        #expect(campaign?.action == .pullModel(
            alias: "qwen3.5-35b-4bit",
            hfRepo: "mlx-community/Qwen3.5-35B-A3B-4bit"
        ))
    }

    @Test("dismissal is namespaced by immutable campaign id")
    func dismissalKey() {
        #expect(Campaign.preview.dismissalKey == "Rapid.campaign.dismissed.model-qwen35-35b-202608")
    }

    @Test("campaign action copy reflects transient download state")
    func actionPresentation() {
        #expect(Campaign.ActionState.idle.isEnabled)
        #expect(Campaign.ActionState.idle.label(fallback: "Download model") == "Download model")
        #expect(!Campaign.ActionState.inProgress.isEnabled)
        #expect(Campaign.ActionState.inProgress.label(fallback: "Download model") == "Downloading…")
        #expect(!Campaign.ActionState.completed.isEnabled)
        #expect(Campaign.ActionState.completed.label(fallback: "Download model") == "Downloaded")
    }
}

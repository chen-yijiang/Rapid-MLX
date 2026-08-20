import SwiftUI

/// A deliberately small, typed slice of the future update-manifest campaign.
/// The server may choose copy, but it cannot ask the app to open an arbitrary
/// URL or execute a command: every action must be a case the client knows.
struct Campaign: Equatable, Sendable {
    enum Kind: String, Sendable { case newModel }
    enum Action: Equatable, Sendable {
        case pullModel(alias: String, hfRepo: String?)
    }

    let id: String
    let kind: Kind
    let title: String
    let body: String
    let actionLabel: String
    let action: Action

    static let preview = Campaign(
        id: "model-qwen35-35b-202608",
        kind: .newModel,
        title: "Qwen3.5 35B is ready",
        body: "A smarter agentic model, tuned for Rapid-MLX. Download it now and keep working locally.",
        actionLabel: "Download model",
        action: .pullModel(
            alias: "qwen3.5-35b-4bit",
            hfRepo: "mlx-community/Qwen3.5-35B-A3B-4bit"
        )
    )

    static func previewFromEnvironment(_ environment: [String: String]) -> Campaign? {
        environment["RAPID_GUI_CAMPAIGN_PREVIEW"] == "1" ? .preview : nil
    }

    static func shouldAcknowledgePull(started: Bool, isDownloading: Bool) -> Bool {
        started || isDownloading
    }

    var dismissalKey: String { "Rapid.campaign.dismissed.\(id)" }
}

struct CampaignBanner: View {
    let campaign: Campaign
    let onAction: (Campaign.Action) -> Void
    let onDismiss: () -> Void

    var body: some View {
        HStack(spacing: RapidTheme.Space.md) {
            Image(systemName: "sparkles")
                .font(.system(size: 16, weight: .semibold))
                .foregroundStyle(RapidTheme.brandAmber)
                .frame(width: 32, height: 32)
                .background(RapidTheme.brandAmberTint, in: RoundedRectangle(cornerRadius: RapidTheme.Radius.row))

            VStack(alignment: .leading, spacing: RapidTheme.Space.xxs) {
                Text(campaign.title)
                    .font(.system(size: 13, weight: .semibold))
                    .foregroundStyle(RapidTheme.textPrimary)
                Text(campaign.body)
                    .font(.system(size: 12))
                    .foregroundStyle(RapidTheme.textSecondary)
                    .lineLimit(2)
            }

            Spacer(minLength: RapidTheme.Space.md)

            Button(campaign.actionLabel) { onAction(campaign.action) }
                .buttonStyle(.rapidPrimaryCompact)
                .accessibilityIdentifier("Campaign.Action")

            Button {
                onDismiss()
            } label: {
                Image(systemName: "xmark")
                    .font(.system(size: 11, weight: .semibold))
                    .frame(width: 24, height: 24)
            }
            .buttonStyle(.plain)
            .help("Dismiss")
            .accessibilityLabel("Dismiss announcement")
            .accessibilityIdentifier("Campaign.Dismiss")
        }
        .padding(.horizontal, RapidTheme.Space.lg)
        .padding(.vertical, RapidTheme.Space.sm)
        .background(RapidTheme.surfaceRaised)
        .overlay(alignment: .bottom) { Rectangle().fill(RapidTheme.hairline).frame(height: 1) }
        .accessibilityElement(children: .contain)
        .accessibilityIdentifier("Campaign.Banner")
    }
}

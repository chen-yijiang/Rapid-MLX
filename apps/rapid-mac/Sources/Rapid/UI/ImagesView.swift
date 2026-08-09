import AppKit
import SwiftUI

/// The Images tab. Deliberately mirrors ``ChatView``: a scrollable results
/// area on top and, at the bottom, the *same* compose box — a `surfaceRaised`
/// rounded field with the model picker + submit button clustered at its
/// bottom-right — so model selection and input feel identical across tabs.
struct ImagesView: View {
    @Bindable var viewModel: ImageGenViewModel
    @Bindable var server: ServerManager

    private let contentMaxWidth: CGFloat = RapidTheme.Layout.contentMaxWidth

    @State private var composeFocusToken = 0
    @State private var pickerHovering = false

    var body: some View {
        VStack(spacing: 0) {
            stageAndHistory
            composer
        }
        .background(RapidTheme.surfaceCanvas)
        .task { await viewModel.refreshCatalog() }
    }

    // MARK: - Stage + history

    private var stageAndHistory: some View {
        VStack(spacing: 12) {
            stage
                .frame(maxWidth: .infinity, maxHeight: .infinity)
            if !viewModel.results.isEmpty {
                filmstrip
            }
        }
        .padding(16)
        .frame(maxWidth: .infinity, maxHeight: .infinity)
    }

    @ViewBuilder
    private var stage: some View {
        ZStack {
            if let active = viewModel.activeImage, let nsImage = NSImage(data: active.pngData) {
                Image(nsImage: nsImage)
                    .resizable()
                    .aspectRatio(contentMode: .fit)
                    .clipShape(RoundedRectangle(cornerRadius: 14))
                    .overlay(
                        RoundedRectangle(cornerRadius: 14).stroke(RapidTheme.hairline, lineWidth: 1)
                    )
                    .overlay(alignment: .topTrailing) { saveOverlay(active) }
                    .accessibilityIdentifier("Images.Stage")
            } else if !viewModel.isGenerating {
                emptyStage
            }

            if viewModel.isGenerating {
                progressHUD.transition(.opacity)
            }
        }
        .frame(maxWidth: .infinity, maxHeight: .infinity)
    }

    @ViewBuilder
    private func saveOverlay(_ image: GeneratedImage) -> some View {
        if !viewModel.isGenerating {
            Button {
                save(image)
            } label: {
                Image(systemName: "square.and.arrow.down")
                    .font(.system(size: 12, weight: .semibold))
                    .frame(width: 28, height: 28)
                    .background(.ultraThinMaterial, in: Circle())
            }
            .buttonStyle(.plain)
            .padding(10)
            .help("Save image")
            .accessibilityIdentifier("Images.Result.Save")
        }
    }

    private var emptyStage: some View {
        VStack(spacing: 12) {
            Image(systemName: "photo.on.rectangle.angled")
                .font(.system(size: 36, weight: .light))
                .foregroundStyle(.tertiary)
            Text("Type a thought below and press Generate.")
                .font(.callout)
                .foregroundStyle(.secondary)
            Text(noModelHint ?? "Runs entirely on this Mac — private, offline, unlimited.")
                .font(.caption)
                .foregroundStyle(.tertiary)
        }
        .multilineTextAlignment(.center)
        .padding(24)
        .frame(maxWidth: .infinity, maxHeight: .infinity)
        .accessibilityIdentifier("Images.EmptyState")
    }

    private var noModelHint: String? {
        guard viewModel.catalogLoaded, viewModel.imageModels.isEmpty else { return nil }
        return "No image models found — install one from Settings → Model Management."
    }

    // MARK: - Progress HUD (the wait, designed)

    private var progressHUD: some View {
        TimelineView(.periodic(from: .now, by: 0.1)) { context in
            let elapsed = viewModel.genStartedAt.map { context.date.timeIntervalSince($0) } ?? 0
            VStack {
                Spacer()
                VStack(alignment: .leading, spacing: 10) {
                    if viewModel.phase == .denoising, let p = viewModel.progress {
                        denoiseBody(p, elapsed: elapsed)
                    } else {
                        preparingBody(elapsed: elapsed)
                    }
                }
                .padding(16)
                .frame(maxWidth: .infinity, alignment: .leading)
                .background(.ultraThinMaterial)
                .clipShape(RoundedRectangle(cornerRadius: 14))
            }
            .padding(2)
        }
    }

    private func preparingBody(elapsed: TimeInterval) -> some View {
        HStack(spacing: 12) {
            ProgressView().controlSize(.small)
            VStack(alignment: .leading, spacing: 2) {
                Text(viewModel.cancelling ? "Stopping…" : "Warming up \(viewModel.selectedDisplayName)…")
                    .font(.system(size: 13, weight: .semibold))
                Text("First run loads the model — this only happens once.")
                    .font(.caption).foregroundStyle(.secondary)
            }
            Spacer()
            clock(elapsed)
            cancelButton
        }
    }

    private func denoiseBody(_ p: ImageClient.ImageProgress, elapsed: TimeInterval) -> some View {
        let total = max(p.total, viewModel.estimatedSteps)
        let fraction = total > 0 ? min(1, Double(p.step) / Double(total)) : 0
        return VStack(alignment: .leading, spacing: 9) {
            HStack {
                Text(viewModel.cancelling
                     ? "Stopping…"
                     : "Step \(max(1, p.step)) / \(total) · denoising")
                    .font(.system(size: 13, weight: .semibold))
                Spacer()
                clock(elapsed)
                cancelButton
            }
            ProgressBar(fraction: fraction).frame(height: 6)
            if let eta = etaText(step: p.step, total: total, elapsed: elapsed) {
                Text(eta).font(.caption).foregroundStyle(.secondary)
            }
        }
    }

    private func clock(_ elapsed: TimeInterval) -> some View {
        Text(String(format: "%.1fs", max(0, elapsed)))
            .font(.system(size: 12, weight: .semibold, design: .monospaced))
            .foregroundStyle(.secondary)
    }

    private var cancelButton: some View {
        Button {
            viewModel.cancel()
        } label: {
            Image(systemName: "xmark")
                .font(.system(size: 11, weight: .bold))
                .frame(width: 22, height: 22)
        }
        .buttonStyle(.plain)
        .background(Color.primary.opacity(0.08))
        .clipShape(Circle())
        .disabled(viewModel.cancelling)
        .help("Cancel")
        .accessibilityIdentifier("Images.Cancel")
    }

    private func etaText(step: Int, total: Int, elapsed: TimeInterval) -> String? {
        guard step > 0, total > step, elapsed > 0 else { return nil }
        let perStep = elapsed / Double(step)
        return "~\(Int((perStep * Double(total - step)).rounded()))s left"
    }

    // MARK: - Filmstrip

    private var filmstrip: some View {
        ScrollView(.horizontal, showsIndicators: false) {
            HStack(spacing: 8) {
                ForEach(viewModel.results) { image in
                    filmstripThumb(image)
                }
            }
            .padding(.vertical, 2)
        }
        .frame(height: 64)
        .accessibilityIdentifier("Images.Gallery")
    }

    private func filmstripThumb(_ image: GeneratedImage) -> some View {
        let selected = viewModel.activeImage?.id == image.id
        return Button {
            viewModel.select(image)
        } label: {
            Group {
                if let nsImage = NSImage(data: image.pngData) {
                    Image(nsImage: nsImage).resizable().aspectRatio(contentMode: .fill)
                } else {
                    Rectangle().fill(RapidTheme.card)
                }
            }
            .frame(width: 56, height: 56)
            .clipShape(RoundedRectangle(cornerRadius: 9))
            .overlay(
                RoundedRectangle(cornerRadius: 9)
                    .stroke(selected ? RapidTheme.brandAmber : RapidTheme.hairline,
                            lineWidth: selected ? 2 : 1)
            )
        }
        .buttonStyle(.plain)
    }

    // MARK: - Composer (mirrors ChatView's compose box)

    private var composer: some View {
        VStack(spacing: RapidTheme.Space.sm) {
            if let error = viewModel.errorMessage {
                InlineNotice(message: error, tone: .error)
                    .frame(maxWidth: contentMaxWidth)
                    .frame(maxWidth: .infinity)
            }
            if viewModel.prompt.isEmpty {
                starters
                    .frame(maxWidth: contentMaxWidth)
                    .frame(maxWidth: .infinity)
            }
            VStack(spacing: RapidTheme.Space.sm - 2) {
                ComposeField(
                    text: $viewModel.prompt,
                    focusToken: composeFocusToken,
                    isStreaming: viewModel.isGenerating,
                    placeholder: "Describe the image you want…",
                    onSubmit: runSubmit,
                    onCancel: { viewModel.cancel() }
                )
                .accessibilityIdentifier("Images.Prompt")
                composerControls
            }
            .padding(.horizontal, RapidTheme.Space.md - 2)
            .padding(.vertical, RapidTheme.Space.sm)
            .background(
                RoundedRectangle(cornerRadius: RapidTheme.Radius.input, style: .continuous)
                    .fill(RapidTheme.surfaceRaised)
            )
            .overlay(
                RoundedRectangle(cornerRadius: RapidTheme.Radius.input, style: .continuous)
                    .strokeBorder(RapidTheme.hairlineStrong, lineWidth: 1)
            )
            .frame(maxWidth: contentMaxWidth)
            .frame(maxWidth: .infinity)
        }
        .padding(.horizontal, RapidTheme.Space.xl)
        .padding(.top, RapidTheme.Space.md)
        .padding(.bottom, RapidTheme.Space.lg)
    }

    /// Bottom row of the compose box: aspect on the left, then the inline
    /// model picker + submit clustered on the right — the same
    /// `model ▾  ⬆` grouping ChatView uses.
    private var composerControls: some View {
        HStack(spacing: RapidTheme.Space.sm) {
            aspectPicker
            Spacer(minLength: 0)
            modelPicker
            sendOrStopButton
        }
    }

    private var starters: some View {
        ScrollView(.horizontal, showsIndicators: false) {
            HStack(spacing: 7) {
                ForEach(ImageGenViewModel.starters, id: \.self) { starter in
                    Button {
                        viewModel.use(starter: starter)
                    } label: {
                        Text(starter)
                            .font(.caption)
                            .lineLimit(1)
                            .padding(.horizontal, 11)
                            .padding(.vertical, 6)
                            .background(RapidTheme.card)
                            .clipShape(Capsule())
                            .overlay(Capsule().stroke(RapidTheme.hairline, lineWidth: 1))
                    }
                    .buttonStyle(.plain)
                    .accessibilityIdentifier("Images.Starter")
                }
            }
        }
    }

    private var aspectPicker: some View {
        HStack(spacing: 4) {
            ForEach(ImageGenViewModel.Aspect.allCases) { ar in
                let on = viewModel.aspect == ar
                Button {
                    viewModel.aspect = ar
                } label: {
                    Text(ar.label)
                        .font(.system(size: 11, weight: .medium))
                        .padding(.horizontal, 8)
                        .padding(.vertical, 5)
                        .background(on ? RapidTheme.hoverFill : Color.clear)
                        .foregroundStyle(on ? Color.primary : Color.secondary)
                        .clipShape(RoundedRectangle(cornerRadius: 6))
                }
                .buttonStyle(.plain)
            }
        }
        .accessibilityIdentifier("Images.Aspect")
    }

    /// The inline model picker — same composer-embedded chip as chat
    /// (``ModelPickerBar`` in `composerStyle`): borderless, a fill on hover,
    /// a cache glyph per row, scaling to any number of image models.
    private var modelPicker: some View {
        Menu {
            if viewModel.imageModels.isEmpty {
                Text(viewModel.catalogLoaded ? "No image models available" : "Loading…")
            } else {
                ForEach(viewModel.imageModels) { entry in
                    Button {
                        viewModel.selectedAlias = entry.alias
                    } label: {
                        Label(
                            modelRowTitle(entry),
                            systemImage: ModelPickerBar.cacheGlyph(cached: entry.cached)
                        )
                    }
                }
            }
        } label: {
            HStack(spacing: 6) {
                Image(systemName: "photo")
                    .foregroundStyle(.secondary)
                    .accessibilityHidden(true)
                Text(viewModel.selectedAlias.isEmpty ? "Choose a model" : viewModel.selectedAlias)
                    .font(RapidFont.secondary)
                    .foregroundStyle(viewModel.selectedAlias.isEmpty ? .secondary : .primary)
                    .lineLimit(1)
                    .truncationMode(.middle)
                Image(systemName: "chevron.up.chevron.down")
                    .font(.system(size: 10, weight: .medium))
                    .foregroundStyle(pickerHovering ? .primary : .secondary)
                    .accessibilityHidden(true)
            }
            .padding(.horizontal, RapidTheme.Space.sm)
            .frame(height: RapidTheme.ControlHeight.small)
            .background(
                RoundedRectangle(cornerRadius: RapidTheme.Radius.row, style: .continuous)
                    .fill(pickerHovering ? RapidTheme.hoverFill : .clear)
            )
            .overlay(
                RoundedRectangle(cornerRadius: RapidTheme.Radius.row, style: .continuous)
                    .strokeBorder(pickerHovering ? RapidTheme.hairlineStrong : .clear, lineWidth: 1)
            )
            .contentShape(RoundedRectangle(cornerRadius: 8, style: .continuous))
        }
        .menuStyle(.button)
        .buttonStyle(.plain)
        .menuIndicator(.hidden)
        .fixedSize()
        .onHover { pickerHovering = $0 }
        .help(viewModel.selectedAlias.isEmpty ? "Choose a model" : "Model: \(viewModel.selectedAlias)")
        .accessibilityIdentifier("Images.ModelPicker")
    }

    private func modelRowTitle(_ entry: ModelEntry) -> String {
        if let size = entry.sizeOnDisk, !size.isEmpty {
            return "\(entry.alias) · \(size)"
        }
        return entry.alias
    }

    /// Submit / stop, styled exactly like ChatView's send button: an amber
    /// disc when there's something to run, a stop disc while generating.
    @ViewBuilder
    private var sendOrStopButton: some View {
        if viewModel.isGenerating {
            Button { viewModel.cancel() } label: {
                Image(systemName: "stop.fill")
                    .font(.system(size: 12, weight: .bold))
                    .foregroundStyle(RapidTheme.sendButtonIcon)
                    .frame(width: 28, height: 28)
                    .background(Circle().fill(RapidTheme.sendButton))
            }
            .buttonStyle(.plain)
            .disabled(viewModel.cancelling)
            .help("Cancel")
            .accessibilityIdentifier("Images.Generate")
        } else {
            Button(action: runSubmit) {
                Image(systemName: "arrow.up")
                    .font(.system(size: 12, weight: .bold))
                    .foregroundStyle(viewModel.canSubmit ? RapidTheme.onBrandPrimary : Color.secondary)
                    .frame(width: 28, height: 28)
                    .background(Circle().fill(viewModel.canSubmit ? RapidTheme.brandPrimary : Color.clear))
                    .overlay(
                        Circle().strokeBorder(
                            viewModel.canSubmit ? .clear : RapidTheme.hairlineStrong, lineWidth: 1)
                    )
            }
            .buttonStyle(.plain)
            .disabled(!viewModel.canSubmit)
            .help("Generate")
            .accessibilityIdentifier("Images.Generate")
        }
    }

    // MARK: - Actions

    private func runSubmit() {
        guard viewModel.canSubmit else { return }
        Task { await viewModel.submit() }
    }

    private func save(_ image: GeneratedImage) {
        let panel = NSSavePanel()
        panel.allowedContentTypes = [.png]
        panel.nameFieldStringValue = "rapid-image.png"
        panel.canCreateDirectories = true
        guard panel.runModal() == .OK, let url = panel.url else { return }
        try? image.pngData.write(to: url)
    }
}

/// A determinate capsule progress bar with the brand fill — the only bar in
/// the app that shows a true diffusion step fraction.
private struct ProgressBar: View {
    let fraction: Double

    var body: some View {
        GeometryReader { geo in
            ZStack(alignment: .leading) {
                Capsule().fill(Color.primary.opacity(0.15))
                Capsule()
                    .fill(RapidTheme.brandAmber)
                    .frame(width: max(0, min(1, fraction)) * geo.size.width)
                    .animation(.easeOut(duration: 0.3), value: fraction)
            }
        }
    }
}

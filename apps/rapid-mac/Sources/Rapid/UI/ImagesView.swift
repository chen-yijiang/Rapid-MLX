import AppKit
import SwiftUI

/// The Images tab — a local text→image surface designed around three jobs:
/// get a good image with the least friction, riff on it, and never feel stuck
/// during a 10–40 s render. Layout: a Fast/Best quality bar, a big focal stage
/// that shows the render's live step progress, a session filmstrip, and a
/// prompt composer with one-tap starters.
struct ImagesView: View {
    @Bindable var viewModel: ImageGenViewModel
    @Bindable var server: ServerManager

    var body: some View {
        VStack(spacing: 0) {
            topBar
            Divider()
            stageAndHistory
            Divider()
            composer
        }
        .background(RapidTheme.surfaceCanvas)
        .task { await viewModel.refreshCatalog() }
    }

    // MARK: - Top bar (model picker + Save)

    private var topBar: some View {
        HStack(spacing: 10) {
            Text("Model").font(.caption).foregroundStyle(.secondary)
            modelPicker
                .frame(minWidth: 220, maxWidth: 320)
            Spacer()
            if let active = viewModel.activeImage, !viewModel.isGenerating {
                Button {
                    save(active)
                } label: {
                    Label("Save", systemImage: "square.and.arrow.down")
                }
                .buttonStyle(.rapidSecondary)
                .accessibilityIdentifier("Images.Result.Save")
            }
        }
        .padding(.horizontal, 16)
        .padding(.vertical, 10)
        .background(RapidTheme.surfaceSidebar)
    }

    /// A dropdown listing every image model — the same shape as the chat
    /// picker (one Menu, a cache glyph per row) so it scales to any number of
    /// models instead of the fixed Fast/Best boxes. Manage/download/delete
    /// live in Settings → Model Management (Image tab).
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
                    .lineLimit(1)
                    .truncationMode(.middle)
                    .foregroundStyle(viewModel.selectedAlias.isEmpty ? .secondary : .primary)
                Spacer(minLength: 4)
                Image(systemName: "chevron.up.chevron.down")
                    .font(.system(size: 10, weight: .medium))
                    .foregroundStyle(.secondary)
                    .accessibilityHidden(true)
            }
            .padding(.horizontal, 10)
            .frame(height: 30)
            .background(
                RoundedRectangle(cornerRadius: 8, style: .continuous)
                    .fill(Color.secondary.opacity(0.06))
            )
            .overlay(
                RoundedRectangle(cornerRadius: 8, style: .continuous)
                    .stroke(RapidTheme.hairline, lineWidth: 1)
            )
            .contentShape(RoundedRectangle(cornerRadius: 8, style: .continuous))
        }
        .menuStyle(.button)
        .buttonStyle(.plain)
        .menuIndicator(.hidden)
        .accessibilityIdentifier("Images.ModelPicker")
    }

    /// "flux2-klein-4b · 4.3 GiB" — alias plus size when known. The green
    /// check / download glyph (from ``ModelPickerBar.cacheGlyph``) carries
    /// installed state, exactly as in the chat picker.
    private func modelRowTitle(_ entry: ModelEntry) -> String {
        if let size = entry.sizeOnDisk, !size.isEmpty {
            return "\(entry.alias) · \(size)"
        }
        return entry.alias
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
                    .accessibilityIdentifier("Images.Stage")
            } else if !viewModel.isGenerating {
                emptyStage
            }

            if viewModel.isGenerating {
                progressHUD
                    .transition(.opacity)
            }
        }
        .frame(maxWidth: .infinity, maxHeight: .infinity)
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
        return "No image models found — install one from the model list."
    }

    // MARK: - Progress HUD (the wait, designed)

    private var progressHUD: some View {
        // A live clock that keeps moving even during the cold model-load phase.
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
            ProgressBar(fraction: fraction)
                .frame(height: 6)
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

    /// ETA from uniform step time; nil until there's a step to extrapolate from.
    private func etaText(step: Int, total: Int, elapsed: TimeInterval) -> String? {
        guard step > 0, total > step, elapsed > 0 else { return nil }
        let perStep = elapsed / Double(step)
        let remaining = perStep * Double(total - step)
        return "~\(Int(remaining.rounded()))s left"
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

    // MARK: - Composer

    private var composer: some View {
        VStack(spacing: 10) {
            if let error = viewModel.errorMessage {
                InlineNotice(message: error, tone: .error)
            }
            if viewModel.prompt.isEmpty {
                starters
            }
            HStack(alignment: .bottom, spacing: 10) {
                TextField("Describe the image you want…", text: $viewModel.prompt, axis: .vertical)
                    .textFieldStyle(.plain)
                    .lineLimit(1...4)
                    .padding(10)
                    .background(RapidTheme.composePill)
                    .clipShape(RoundedRectangle(cornerRadius: 10))
                    .accessibilityIdentifier("Images.Prompt")
                    .onSubmit(runSubmit)

                aspectPicker

                Button(action: runSubmit) {
                    if viewModel.isGenerating {
                        ProgressView().controlSize(.small)
                    } else {
                        Text("Generate")
                    }
                }
                .buttonStyle(.rapidPrimary)
                .disabled(!viewModel.canSubmit)
                .accessibilityIdentifier("Images.Generate")
            }
        }
        .padding(12)
        .background(RapidTheme.surfaceSidebar)
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
                        .padding(.horizontal, 9)
                        .padding(.vertical, 7)
                        .background(on ? RapidTheme.card : Color.clear)
                        .foregroundStyle(on ? Color.primary : Color.secondary)
                        .clipShape(RoundedRectangle(cornerRadius: 6))
                }
                .buttonStyle(.plain)
            }
        }
        .padding(3)
        .background(RapidTheme.composePill)
        .clipShape(RoundedRectangle(cornerRadius: 9))
        .accessibilityIdentifier("Images.Aspect")
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

/// A determinate capsule progress bar with the brand fill. Kept tiny and
/// local — the only progress bar in the app that shows a true diffusion
/// step fraction.
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

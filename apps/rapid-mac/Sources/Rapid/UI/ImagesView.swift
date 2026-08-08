import AppKit
import SwiftUI

/// The Images tab — a dedicated text→image / image-edit surface, decoupled
/// from chat (rapid-mlx serves one model per process, so image generation
/// runs against an image-gen alias rather than the chat model). Layout
/// mirrors ``ChatView``: a results area on top, a compose bar on the bottom.
struct ImagesView: View {
    @Bindable var viewModel: ImageGenViewModel
    @Bindable var server: ServerManager

    /// Curated output sizes. Kept small (all multiples of 16, within the
    /// engine's 256–2048 bounds) so the picker can't produce a 400.
    private let sizeOptions = ["512x512", "768x768", "1024x1024", "1024x768", "768x1024"]

    var body: some View {
        VStack(spacing: 0) {
            gallery
            Divider()
            composeBar
        }
        .background(RapidTheme.surfaceCanvas)
        .task { await viewModel.refreshCatalog() }
    }

    // MARK: - Gallery

    @ViewBuilder
    private var gallery: some View {
        if viewModel.results.isEmpty {
            EmptyState(
                symbol: "photo.on.rectangle.angled",
                title: "Generate an image",
                message: "Describe what you want to see, pick an image model, and press Generate.",
                hint: viewModel.imageModels.isEmpty && viewModel.catalogLoaded
                    ? "No image models found — install one from the model list."
                    : nil
            )
            .frame(maxWidth: .infinity, maxHeight: .infinity)
            .accessibilityIdentifier("Images.EmptyState")
        } else {
            ScrollView {
                LazyVGrid(
                    columns: [GridItem(.adaptive(minimum: 220), spacing: 16)],
                    spacing: 16
                ) {
                    ForEach(viewModel.results) { image in
                        resultCard(image)
                    }
                }
                .padding(16)
            }
            .accessibilityIdentifier("Images.Gallery")
        }
    }

    private func resultCard(_ image: GeneratedImage) -> some View {
        VStack(alignment: .leading, spacing: 8) {
            if let nsImage = NSImage(data: image.pngData) {
                Image(nsImage: nsImage)
                    .resizable()
                    .aspectRatio(contentMode: .fit)
                    .frame(maxWidth: .infinity)
                    .clipShape(RoundedRectangle(cornerRadius: 10))
                    .overlay(
                        RoundedRectangle(cornerRadius: 10)
                            .stroke(RapidTheme.hairline, lineWidth: 1)
                    )
            }
            Text(image.prompt)
                .font(.caption)
                .foregroundStyle(.secondary)
                .lineLimit(2)
            HStack(spacing: 8) {
                Button("Edit") { viewModel.beginEdit(image) }
                    .buttonStyle(.rapidSecondary)
                    .accessibilityIdentifier("Images.Result.Edit")
                Button("Save") { save(image) }
                    .buttonStyle(.rapidSecondary)
                    .accessibilityIdentifier("Images.Result.Save")
                Spacer()
                if image.isEdit {
                    Text("edited")
                        .font(.caption2)
                        .foregroundStyle(RapidTheme.brandAmber)
                }
            }
        }
        .padding(10)
        .background(RapidTheme.card)
        .clipShape(RoundedRectangle(cornerRadius: 12))
    }

    // MARK: - Compose bar

    private var composeBar: some View {
        VStack(spacing: 8) {
            if viewModel.editSource != nil {
                editingBanner
            }
            if let error = viewModel.errorMessage {
                InlineNotice(message: error, tone: .error)
            }
            HStack(spacing: 12) {
                modelPicker
                sizePicker
                Spacer()
            }
            HStack(spacing: 10) {
                TextField(
                    viewModel.editSource != nil
                        ? "Describe the change…"
                        : "Describe the image…",
                    text: $viewModel.prompt,
                    axis: .vertical
                )
                .textFieldStyle(.plain)
                .lineLimit(1...4)
                .padding(10)
                .background(RapidTheme.composePill)
                .clipShape(RoundedRectangle(cornerRadius: 10))
                .accessibilityIdentifier("Images.Prompt")
                .onSubmit { runSubmit() }

                Button(action: runSubmit) {
                    if viewModel.isGenerating {
                        ProgressView().controlSize(.small)
                    } else {
                        Text(viewModel.editSource != nil ? "Edit" : "Generate")
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

    private var editingBanner: some View {
        HStack(spacing: 8) {
            Image(systemName: "wand.and.stars")
                .foregroundStyle(RapidTheme.brandAmber)
            Text("Editing an image")
                .font(.callout)
            Spacer()
            Button("Cancel") { viewModel.cancelEdit() }
                .buttonStyle(.rapidSecondary)
                .accessibilityIdentifier("Images.CancelEdit")
        }
        .padding(.horizontal, 4)
    }

    private var modelPicker: some View {
        Picker("Model", selection: $viewModel.selectedAlias) {
            ForEach(viewModel.imageModels) { entry in
                Text(entry.cached ? "\(entry.alias) ✓" : entry.alias)
                    .tag(entry.alias)
            }
        }
        .labelsHidden()
        .frame(maxWidth: 260)
        .accessibilityIdentifier("Images.ModelPicker")
    }

    private var sizePicker: some View {
        Picker("Size", selection: $viewModel.size) {
            ForEach(sizeOptions, id: \.self) { Text($0).tag($0) }
        }
        .labelsHidden()
        .frame(maxWidth: 120)
        .accessibilityIdentifier("Images.SizePicker")
    }

    // MARK: - Actions

    private func runSubmit() {
        guard viewModel.canSubmit else { return }
        Task { await viewModel.submit() }
    }

    /// Write the PNG to a user-chosen location via the standard save panel.
    private func save(_ image: GeneratedImage) {
        let panel = NSSavePanel()
        panel.allowedContentTypes = [.png]
        panel.nameFieldStringValue = "rapid-image.png"
        panel.canCreateDirectories = true
        guard panel.runModal() == .OK, let url = panel.url else { return }
        try? image.pngData.write(to: url)
    }
}

import AppKit
import SwiftUI

/// Hosts the process-switch confirmation on whichever Rapid window is active.
/// The manager is app-scoped, so anchoring this only in ContentView strands a
/// Settings-initiated restart when the main window is closed.
private struct ActiveRequestSwitchAlertModifier: ViewModifier {
    @Environment(ServerManager.self) private var server
    @State private var hostWindow: NSWindow?

    func body(content: Content) -> some View {
        let displayedWarning = server.pendingActiveRequestSwitch
        content
            .background {
                WindowAccessor { window in hostWindow = window }
                    .frame(width: 0, height: 0)
            }
            .alert(
                "Switch models?",
                isPresented: Binding(
                    get: {
                        displayedWarning != nil
                            && (hostWindow?.isKeyWindow == true || NSApp.keyWindow == nil)
                    },
                    set: {
                        if !$0, let warning = displayedWarning {
                            server.cancelActiveRequestSwitch(warning)
                        }
                    }
                ),
                presenting: displayedWarning
            ) { warning in
                Button("Cancel", role: .cancel) {
                    server.cancelActiveRequestSwitch(warning)
                }
                .accessibilityIdentifier("ActiveRequestSwitch.Cancel")
                Button("Switch model", role: .destructive) {
                    server.confirmActiveRequestSwitch(warning)
                }
                .accessibilityIdentifier("ActiveRequestSwitch.Confirm")
            } message: { warning in
                if warning.activeRequests == 0 {
                    Text("Switching from \(warning.currentAlias) to \(warning.targetAlias) briefly stops the current server. New requests may be interrupted during the handoff.")
                } else if let count = warning.activeRequests {
                    Text("\(count) active request\(count == 1 ? " is" : "s are") using \(warning.currentAlias). Switching to \(warning.targetAlias) will interrupt \(count == 1 ? "it" : "them").")
                } else {
                    Text("Rapid couldn't verify whether requests are active. Switching from \(warning.currentAlias) to \(warning.targetAlias) may interrupt clients using the server.")
                }
            }
    }
}

extension View {
    func activeRequestSwitchAlert() -> some View {
        modifier(ActiveRequestSwitchAlertModifier())
    }
}

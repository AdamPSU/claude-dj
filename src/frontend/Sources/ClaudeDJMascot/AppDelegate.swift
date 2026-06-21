import AppKit
import SwiftUI

/// Wires together the overlay panel, the crawl animation, and the system
/// sleep/wake lifecycle.
///
/// Behavior:
/// - On launch (and on every wake from sleep): the box is stationary at the
///   left end of the Dock's top edge.
/// - On click: the box crawls left -> right -> left, repeating forever.
/// - On sleep: the crawl stops; on the next wake it is reset to stationary,
///   awaiting a fresh click.
final class AppDelegate: NSObject, NSApplicationDelegate {
    private var panel: OverlayPanel!
    private var crawler: Crawler!

    // Tunables.
    private let boxSize: CGFloat = 64        // placeholder box edge length (points)
    private let crawlSpeed: CGFloat = 220    // points per second

    func applicationDidFinishLaunching(_ notification: Notification) {
        requestAccessibilityIfNeeded()

        panel = OverlayPanel(size: boxSize)
        crawler = Crawler(window: panel, boxSize: boxSize, speed: crawlSpeed)

        let view = MascotView(size: boxSize) { [weak self] in
            self?.crawler.start()
        }
        panel.contentView = NSHostingView(rootView: view)

        crawler.placeAtStart()
        panel.orderFrontRegardless()

        let workspaceCenter = NSWorkspace.shared.notificationCenter
        workspaceCenter.addObserver(self, selector: #selector(handleWake),
                                    name: NSWorkspace.didWakeNotification, object: nil)
        workspaceCenter.addObserver(self, selector: #selector(handleSleep),
                                    name: NSWorkspace.willSleepNotification, object: nil)

        NotificationCenter.default.addObserver(self, selector: #selector(handleScreenChange),
                                               name: NSApplication.didChangeScreenParametersNotification, object: nil)
    }

    @objc private func handleWake() {
        // Back to stationary top-left; wait for the user to click again.
        crawler.reset()
    }

    @objc private func handleSleep() {
        crawler.stop()
    }

    @objc private func handleScreenChange() {
        // Dock size/position may have changed; re-park if we're idle.
        if !crawler.isCrawling {
            crawler.placeAtStart()
        }
    }

    /// Prompts for Accessibility access, which is required to read the exact Dock
    /// pill rect. If not granted, the app still runs using a full-width fallback.
    private func requestAccessibilityIfNeeded() {
        // Literal value of `kAXTrustedCheckOptionPrompt` ("AXTrustedCheckOptionPrompt"),
        // used directly to avoid CFString-import ambiguity across SDK versions.
        let promptKey = "AXTrustedCheckOptionPrompt"
        let options = [promptKey: true] as CFDictionary
        let trusted = AXIsProcessTrustedWithOptions(options)
        if !trusted {
            NSLog("[ClaudeDJMascot] Accessibility not granted — using full-screen-width fallback. "
                + "Grant access in System Settings > Privacy & Security > Accessibility, then relaunch "
                + "for exact Dock-pill crawling.")
        }
    }
}

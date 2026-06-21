import AppKit

/// A borderless, transparent panel that floats just above the system Dock and
/// shows on every Space.
///
/// We use an `NSPanel` (not a plain `NSWindow`) with `.nonactivatingPanel` so
/// that clicking the mascot does NOT steal focus from whatever app the user is
/// currently working in. `canBecomeKey` is still overridden to `true` so the
/// SwiftUI tap gesture inside the hosted view receives mouse events.
final class OverlayPanel: NSPanel {
    init(size: CGFloat) {
        super.init(
            contentRect: NSRect(x: 0, y: 0, width: size, height: size),
            styleMask: [.borderless, .nonactivatingPanel],
            backing: .buffered,
            defer: false
        )

        isFloatingPanel = true

        // Sit one level above the Dock so the mascot renders on top of it
        // rather than behind it.
        level = NSWindow.Level(rawValue: Int(CGWindowLevelForKey(.dockWindow)) + 1)

        // Visible on all Spaces, doesn't move when Spaces switch, and stays out
        // of the Cmd-Tab / window-cycling lists.
        collectionBehavior = [.canJoinAllSpaces, .stationary, .fullScreenAuxiliary, .ignoresCycle]

        backgroundColor = .clear
        isOpaque = false
        hasShadow = false
        isMovableByWindowBackground = false
        isMovable = false
        ignoresMouseEvents = false
        hidesOnDeactivate = false
    }

    // Required so the hosted SwiftUI view can receive the tap that starts the crawl.
    override var canBecomeKey: Bool { true }
    override var canBecomeMain: Bool { false }
}

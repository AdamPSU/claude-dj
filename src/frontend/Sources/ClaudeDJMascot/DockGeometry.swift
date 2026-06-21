import AppKit
import ApplicationServices

/// Resolves where the mascot should travel: the horizontal span of the Dock's
/// top edge, expressed in Cocoa (bottom-left origin) screen coordinates that
/// `NSWindow.setFrameOrigin` understands.
///
/// Primary path: query the Dock process via the Accessibility API for the exact
/// centered Dock "pill" rect. Fallback (when Accessibility is not granted or the
/// pill can't be found): the full width of the main screen at the Dock's top
/// edge, derived from `NSScreen` frames (no permissions required).
///
/// Assumes a bottom, always-visible Dock (per current product decision).
struct DockGeometry {
    /// Minimum window origin x (left end of the crawl).
    let left: CGFloat
    /// Maximum window origin x (right end of the crawl, already inset by box width).
    let right: CGFloat
    /// Cocoa y for the window origin so the box's bottom rests on the Dock's top edge.
    let topEdgeY: CGFloat

    /// `true` when the values came from the real Dock pill, `false` for the fallback.
    let usedAccessibility: Bool

    static func current(boxSize: CGFloat) -> DockGeometry? {
        if let pill = dockPillRectCocoa() {
            let left = pill.minX
            let right = max(pill.maxX - boxSize, left)
            return DockGeometry(left: left, right: right, topEdgeY: pill.maxY, usedAccessibility: true)
        }
        return fallback(boxSize: boxSize)
    }

    // MARK: - Accessibility path

    /// The Dock pill rect converted to Cocoa coordinates, or nil if unavailable.
    private static func dockPillRectCocoa() -> CGRect? {
        guard AXIsProcessTrusted() else { return nil }
        guard let dock = NSWorkspace.shared.runningApplications
            .first(where: { $0.bundleIdentifier == "com.apple.dock" }) else { return nil }

        let axApp = AXUIElementCreateApplication(dock.processIdentifier)
        guard let list = firstChildWithListRole(of: axApp) else { return nil }
        guard let quartzRect = frame(of: list) else { return nil }
        return cocoaRect(fromQuartz: quartzRect)
    }

    private static func firstChildWithListRole(of element: AXUIElement) -> AXUIElement? {
        var childrenRef: CFTypeRef?
        guard AXUIElementCopyAttributeValue(element, kAXChildrenAttribute as CFString, &childrenRef) == .success,
              let children = childrenRef as? [AXUIElement] else { return nil }

        for child in children {
            var roleRef: CFTypeRef?
            guard AXUIElementCopyAttributeValue(child, kAXRoleAttribute as CFString, &roleRef) == .success else { continue }
            if let role = roleRef as? String, role == (kAXListRole as String) {
                return child
            }
        }
        return nil
    }

    /// Position + size of an Accessibility element as a Quartz (top-left origin,
    /// y-down) rect.
    private static func frame(of element: AXUIElement) -> CGRect? {
        var posRef: CFTypeRef?
        var sizeRef: CFTypeRef?
        guard AXUIElementCopyAttributeValue(element, kAXPositionAttribute as CFString, &posRef) == .success,
              AXUIElementCopyAttributeValue(element, kAXSizeAttribute as CFString, &sizeRef) == .success,
              let posValue = posRef, let sizeValue = sizeRef else { return nil }

        var pos = CGPoint.zero
        var size = CGSize.zero
        guard AXValueGetValue(posValue as! AXValue, .cgPoint, &pos),
              AXValueGetValue(sizeValue as! AXValue, .cgSize, &size) else { return nil }

        return CGRect(origin: pos, size: size)
    }

    /// Converts a Quartz (top-left origin, y-down) rect to a Cocoa (bottom-left
    /// origin, y-up) rect. The reference height is the primary screen — the one
    /// whose Cocoa origin is (0, 0).
    private static func cocoaRect(fromQuartz q: CGRect) -> CGRect {
        let zeroScreenHeight = NSScreen.screens.first?.frame.height ?? NSScreen.main?.frame.height ?? 0
        return CGRect(x: q.minX, y: zeroScreenHeight - q.maxY, width: q.width, height: q.height)
    }

    // MARK: - Fallback path (no permissions)

    private static func fallback(boxSize: CGFloat) -> DockGeometry? {
        guard let screen = NSScreen.main else { return nil }
        let frame = screen.frame
        let visible = screen.visibleFrame
        // For a bottom Dock, the top of the Dock is the bottom of the visible area.
        let topEdgeY = visible.minY
        let left = frame.minX
        let right = max(frame.maxX - boxSize, left)
        return DockGeometry(left: left, right: right, topEdgeY: topEdgeY, usedAccessibility: false)
    }
}

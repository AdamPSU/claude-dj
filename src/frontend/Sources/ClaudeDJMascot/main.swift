import AppKit

// Entry point. This is an accessory (agent) app: no Dock icon of its own and no
// menu bar. The visible surface is a single borderless, transparent overlay
// panel that crawls along the top edge of the system Dock.
//
// The `delegate` constant is held for the lifetime of the program so the
// AppDelegate (and everything it owns) is not deallocated.
let app = NSApplication.shared
let delegate = AppDelegate()
app.delegate = delegate
app.setActivationPolicy(.accessory)
app.run()

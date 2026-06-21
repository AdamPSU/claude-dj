# ClaudeDJ Mascot (SwiftUI frontend)

A native macOS overlay: a small box that sits on the **top edge of the Dock**.

- **On launch / after every wake from sleep:** stationary at the **left end** of the Dock's top edge.
- **On click:** crawls left → right → left, repeating forever.
- **On sleep:** stops; resets to stationary on the next wake (click again to resume).

Currently the mascot is a **placeholder black box**. The hosting window is
transparent, so swapping in an animated asset later requires changing only
`MascotView.swift` (replace the `Rectangle` with your `Image`/animated view).

This is the **only** Swift in the project. The backend stays in Python; future
backends will also be Python. There is **no** frontend↔backend bridge yet — this
component is purely visual.

## Requirements

- macOS 13+
- Swift toolchain (Xcode or Command Line Tools): `swift --version`

## Run

```sh
cd src/frontend
swift run
```

The app is an **accessory/agent app**: it has no Dock icon of its own and no
menu bar. Quit it with `Ctrl-C` in the terminal where `swift run` is running.

## Accessibility permission (for exact Dock-pill crawling)

To crawl across the **actual centered Dock width**, the app reads the Dock's
geometry via the Accessibility API. On first launch you'll be prompted; grant the
running binary (or your terminal) access under:

**System Settings → Privacy & Security → Accessibility**

then relaunch. Without it, the app still runs but falls back to crawling across
the **full screen width** at the Dock's top-edge height (no permission needed).

> Note: with `swift run`, the granted binary lives under `.build/`. A `release`
> build path is more stable for repeated grants: `swift run -c release`.

## Assumptions / scope (current)

- Dock is at the **bottom** and **always visible** (not auto-hidden).
- **Main screen** only (not multi-monitor).
- A single click starts the perpetual crawl; only sleep stops it.

## Tunables

In `AppDelegate.swift`: `boxSize` (box edge length) and `crawlSpeed` (points/sec).

## File map

| File | Responsibility |
|------|----------------|
| `main.swift` | Entry point; configures the app as an accessory app. |
| `AppDelegate.swift` | Wires panel + crawler + sleep/wake lifecycle. |
| `OverlayPanel.swift` | Borderless, transparent, above-Dock, all-Spaces panel. |
| `DockGeometry.swift` | Resolves the Dock pill rect (Accessibility) with NSScreen fallback. |
| `Crawler.swift` | Timer-driven triangle-wave motion of the window origin. |
| `MascotView.swift` | The placeholder black box (swap point for the real asset). |

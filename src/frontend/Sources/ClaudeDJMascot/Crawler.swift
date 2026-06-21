import AppKit
import QuartzCore

/// Drives the mascot's motion by animating the overlay window's frame origin
/// along the Dock's top edge.
///
/// Motion is a triangle wave (left -> right -> left, repeating) computed from
/// elapsed time, so it is frame-rate independent and loops forever until
/// `stop()`/`reset()` is called.
final class Crawler {
    private weak var window: NSWindow?
    private var timer: Timer?
    private var startTime: CFTimeInterval = 0

    private var left: CGFloat = 0
    private var right: CGFloat = 0
    private var edgeY: CGFloat = 0

    private(set) var isCrawling = false

    private let boxSize: CGFloat
    /// Crawl speed in points per second.
    private let speed: CGFloat

    init(window: NSWindow, boxSize: CGFloat, speed: CGFloat) {
        self.window = window
        self.boxSize = boxSize
        self.speed = speed
    }

    /// Park the box, stationary, at the left end of the Dock's top edge.
    func placeAtStart() {
        guard let geom = DockGeometry.current(boxSize: boxSize) else { return }
        left = geom.left
        right = geom.right
        edgeY = geom.topEdgeY
        window?.setFrameOrigin(CGPoint(x: left, y: edgeY))
    }

    /// Begin the perpetual back-and-forth crawl. No-op if already crawling.
    func start() {
        guard !isCrawling else { return }
        guard let geom = DockGeometry.current(boxSize: boxSize) else { return }
        left = geom.left
        right = geom.right
        edgeY = geom.topEdgeY

        startTime = CACurrentMediaTime()
        isCrawling = true

        let timer = Timer(timeInterval: 1.0 / 120.0, repeats: true) { [weak self] _ in
            self?.tick()
        }
        // .common so the animation keeps running during menu tracking, etc.
        RunLoop.main.add(timer, forMode: .common)
        self.timer = timer
    }

    /// Stop animating but leave the box wherever it is.
    func stop() {
        timer?.invalidate()
        timer = nil
        isCrawling = false
    }

    /// Stop and return to the stationary start position (used on wake).
    func reset() {
        stop()
        placeAtStart()
    }

    private func tick() {
        let span = max(right - left, 1)
        let elapsed = CACurrentMediaTime() - startTime
        // Triangle wave over a period of 2*span worth of travel.
        let dist = (elapsed * Double(speed)).truncatingRemainder(dividingBy: Double(2 * span))
        let offset = dist <= Double(span) ? CGFloat(dist) : CGFloat(2 * Double(span) - dist)
        window?.setFrameOrigin(CGPoint(x: left + offset, y: edgeY))
    }
}

import SwiftUI

/// The user-facing surface. For now this is a placeholder black box.
///
/// To swap in the final animated asset later: replace the `Rectangle` with an
/// `Image`/animated view. The hosting window is already transparent
/// (`backgroundColor = .clear`, `isOpaque = false`), so any transparent PNG /
/// GIF / Lottie-style asset will composite correctly over the Dock — no window
/// changes required.
struct MascotView: View {
    let size: CGFloat
    let onTap: () -> Void

    var body: some View {
        Rectangle()
            .fill(Color.black)
            .frame(width: size, height: size)
            // contentShape makes the whole frame tappable even if the fill
            // becomes partially transparent once the real asset is dropped in.
            .contentShape(Rectangle())
            .onTapGesture { onTap() }
    }
}

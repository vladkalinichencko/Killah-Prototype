import AppKit

class CustomWindow: NSWindow {
    override init(contentRect: NSRect, styleMask style: NSWindow.StyleMask, backing backingStoreType: NSWindow.BackingStoreType, defer flag: Bool) {
        super.init(contentRect: contentRect, styleMask: [.titled, .closable, .miniaturizable, .resizable, .fullSizeContentView], backing: backingStoreType, defer: flag)
        self.titlebarAppearsTransparent = true
        self.isMovableByWindowBackground = true
        self.backgroundColor = .clear
        self.isOpaque = false
        self.hasShadow = true

        // Add blur effect to the entire window
        let visualEffectView = NSVisualEffectView(frame: self.contentView?.bounds ?? .zero)
        visualEffectView.autoresizingMask = [.width, .height]
        visualEffectView.material = .underWindowBackground
        visualEffectView.state = .active
        visualEffectView.blendingMode = .behindWindow
        self.contentView?.addSubview(visualEffectView, positioned: .below, relativeTo: nil)
    }
}

class CustomWindowController: NSWindowController {
    convenience init(rootView: NSView) {
        let window = CustomWindow(
            contentRect: NSRect(x: 0, y: 0, width: 900, height: 700),
            styleMask: [.titled, .closable, .miniaturizable, .resizable, .fullSizeContentView],
            backing: .buffered,
            defer: false
        )
        self.init(window: window)
        window.contentView?.addSubview(rootView)
        rootView.translatesAutoresizingMaskIntoConstraints = false
        NSLayoutConstraint.activate([
            rootView.topAnchor.constraint(equalTo: window.contentView!.topAnchor),
            rootView.bottomAnchor.constraint(equalTo: window.contentView!.bottomAnchor),
            rootView.leadingAnchor.constraint(equalTo: window.contentView!.leadingAnchor),
            rootView.trailingAnchor.constraint(equalTo: window.contentView!.trailingAnchor)
        ])
    }
}

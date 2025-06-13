import SwiftUI
import AppKit
import QuartzCore

/// A component that manages a custom interactive caret and its associated menu in a text view.
class SmartCaret {
    weak var textView: CustomInlineNSTextView?
    
    /// Core animation layer representing the caret.
    private var interactiveCursorLayer: CALayer?
    private var cursorTrackingArea: NSTrackingArea?
    private var menuHostingView: NSHostingView<CursorMenu>?
    private var cursorManager = InteractiveCursorManager()
    private var cursorUpdateTimer: Timer?
    
    init(textView: CustomInlineNSTextView) {
        self.textView = textView
        setup()
    }
    
    /// Setup the caret: hide default cursor, create layer, tracking and updates.
    private func setup() {
        guard let view = textView else { return }
        view.wantsLayer = true
        view.insertionPointColor = .clear
        setupCursorLayer(in: view)
        setupCursorTracking(in: view)
        startCursorUpdates()
    }
    
    /// Create and add the caret layer to the text view's layer.
    private func setupCursorLayer(in view: CustomInlineNSTextView) {
        let layer = CALayer()
        layer.backgroundColor = NSColor.systemBlue.cgColor
        layer.cornerRadius = 1.0
        layer.opacity = 1.0
        view.layer?.addSublayer(layer)
        // Sync layer coordinate orientation with the view's flipped coordinate system
        if let rootLayer = view.layer {
            rootLayer.isGeometryFlipped = view.isFlipped
        }
        // Make layer coordinate system match flipped NSTextView coordinates
        view.layer?.isGeometryFlipped = true
        interactiveCursorLayer = layer
        updateAppearance()
    }
    
    /// Setup a tracking area for hover and click events.
    func setupCursorTracking(in view: CustomInlineNSTextView) {
        if let existing = cursorTrackingArea {
            view.removeTrackingArea(existing)
        }
        let area = NSTrackingArea(rect: view.bounds,
                                  options: [.mouseEnteredAndExited, .mouseMoved, .activeWhenFirstResponder],
                                  owner: view,
                                  userInfo: nil)
        view.addTrackingArea(area)
        cursorTrackingArea = area
    }
    
    /// Start a timer to update the caret position 60 FPS.
    private func startCursorUpdates() {
        cursorUpdateTimer = Timer.scheduledTimer(withTimeInterval: 1/60.0, repeats: true) { [weak self] _ in
            self?.updatePosition()
        }
    }
    
    /// Update the caret layer frame based on the current selection using AppKit's firstRect API.
    private func updatePosition() {
        guard let view = textView,
              let layer = interactiveCursorLayer else { return }

        // Get the caret rectangle in screen coordinates
        let selRange = view.selectedRange
        let screenRect = view.firstRect(forCharacterRange: selRange, actualRange: nil)
        guard let window = view.window else { return }
        let windowRect = window.convertFromScreen(screenRect)
        let viewRect = view.convert(windowRect, from: nil)

        let width = cursorManager.isHovered ? 3.0 : 2.0
        let frame = CGRect(x: viewRect.minX,
                           y: viewRect.minY,
                           width: width,
                           height: viewRect.height)
        CATransaction.begin()
        CATransaction.setAnimationDuration(0.1)
        layer.frame = frame
        CATransaction.commit()
    }
    
    /// Update caret color and scale on hover.
    private func updateAppearance() {
        guard let layer = interactiveCursorLayer else { return }
        CATransaction.begin()
        CATransaction.setAnimationDuration(0.2)
        if cursorManager.isHovered {
            layer.backgroundColor = NSColor.systemOrange.cgColor
            layer.transform = CATransform3DMakeScale(1.2, 1.0, 1.0)
        } else {
            layer.backgroundColor = NSColor.systemBlue.cgColor
            layer.transform = CATransform3DIdentity
        }
        CATransaction.commit()
    }
    
    /// Handle a mouse down event in the text view.
    func handleMouseDown(event: NSEvent) {
        guard let view = textView, let layer = interactiveCursorLayer else { return }
        let local = view.convert(event.locationInWindow, from: nil)
        let clickArea = layer.frame.insetBy(dx: -15, dy: -10)
        if clickArea.contains(local) {
            // Toggle menu visibility
            if cursorManager.isMenuVisible {
                hideMenu()
            } else {
                showMenu(event: event)
            }
        } else {
            hideMenu()
        }
    }
    
    /// Show the SwiftUI menu at the caret location.
    private func showMenu(event: NSEvent) {
        guard let view = textView else { return }
        let menuType: CursorMenuType
        if event.modifierFlags.contains(.command) {
            menuType = .aiActions
        } else if event.modifierFlags.contains(.option) {
            menuType = .audioRecording
        } else if event.modifierFlags.contains(.shift) {
            menuType = .textFormatting
        } else {
            menuType = .promptInput
        }
        hideMenu()
        let menu = CursorMenu(menuType: menuType) { [weak self] in self?.hideMenu() }
        let host = NSHostingView(rootView: menu)
        host.translatesAutoresizingMaskIntoConstraints = false
        view.superview?.addSubview(host)
        let posInWindow = view.convert(layerPosition(), to: nil)
        host.frame = CGRect(x: posInWindow.x + 20,
                            y: posInWindow.y - 50,
                            width: 280, height: 200)
        host.alphaValue = 0.0
        NSAnimationContext.runAnimationGroup { ctx in
            ctx.duration = 0.3
            ctx.allowsImplicitAnimation = true
            host.alphaValue = 1.0
        }
        menuHostingView = host
        cursorManager.showMenu(type: menuType, at: layerPosition())
    }
    
    /// Hide and remove the menu.
    private func hideMenu() {
        guard let host = menuHostingView else { return }
        NSAnimationContext.runAnimationGroup({ ctx in
            ctx.duration = 0.2
            ctx.allowsImplicitAnimation = true
            host.alphaValue = 0.0
        }) {
            host.removeFromSuperview()
        }
        cursorManager.hideMenu()
        menuHostingView = nil
    }
    
    /// Check for hover updates on mouse moved.
    func handleMouseMoved(event: NSEvent) {
        guard let view = textView else { return }
        let local = view.convert(event.locationInWindow, from: nil)
        guard let layer = interactiveCursorLayer else { return }
        let hoverArea = layer.frame.insetBy(dx: -10, dy: -5)
        let was = cursorManager.isHovered
        cursorManager.isHovered = hoverArea.contains(local)
        if was != cursorManager.isHovered {
            updateAppearance()
        }
    }
    
    /// Helper to get current caret layer position in view coordinates.
    private func layerPosition() -> CGPoint {
        return interactiveCursorLayer?.frame.origin ?? .zero
    }
    
    deinit {
        cursorUpdateTimer?.invalidate()
        if let area = cursorTrackingArea, let view = textView {
            view.removeTrackingArea(area)
        }
        hideMenu()
    }
}

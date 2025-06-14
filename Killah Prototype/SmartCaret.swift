import SwiftUI
import AppKit
import QuartzCore

// MARK: - Audio Waveform Component

struct AudioWaveform: View {
    @State private var isAnimating = false
    
    var body: some View {
        HStack(spacing: 1) {
            ForEach(0..<20, id: \.self) { index in
                RoundedRectangle(cornerRadius: 0.5)
                    .fill(Color.blue.opacity(0.6))
                    .frame(width: 2, height: waveHeight(for: index))
                    .scaleEffect(y: isAnimating ? 1.0 : 0.3)
                    .animation(
                        .easeInOut(duration: 0.6)
                        .repeatForever(autoreverses: true)
                        .delay(Double(index) * 0.05),
                        value: isAnimating
                    )
            }
        }
        .onAppear {
            isAnimating = true
        }
    }
    
    private func waveHeight(for index: Int) -> CGFloat {
        let heights: [CGFloat] = [4, 8, 12, 16, 20, 16, 12, 8, 4, 6, 10, 14, 18, 14, 10, 6, 4, 8, 12, 8]
        return heights[index % heights.count]
    }
}

// MARK: - Interactive Cursor SwiftUI Components

enum CursorMenuState {
    case inactive
    case promptHover
    case recordHover
    case promptTap
    case recordTap
    case promptEnter
    case promptAnswer
    case promptHistory
    case promptLarge
    case recordLarge
}

struct CursorMenu: View {
    var state: CursorMenuState
    @State private var inputText: String = ""
    @State private var isHovered: Bool = false
    @State private var isVisible: Bool = false
    
    var body: some View {
        Group {
            switch state {
            case .inactive:
                Rectangle()
                    .fill(Color.red)
                    .frame(width: 2, height: 24)
                    .opacity(isVisible ? 1.0 : 0.0)
                    .onAppear {
                        withAnimation(.easeInOut(duration: 0.3)) {
                            isVisible = true
                        }
                    }
            case .promptHover:
                HStack(spacing: 4) {
                    Circle()
                        .fill(Color.red)
                        .frame(width: 20, height: 20)
                    RoundedRectangle(cornerRadius: 6)
                        .fill(Color.blue.opacity(0.2))
                        .stroke(Color.blue.opacity(0.3), lineWidth: 1)
                        .frame(width: 80, height: 24)
                        .scaleEffect(x: isVisible ? 1.0 : 0.0, y: 1.0, anchor: .leading)
                        .opacity(isVisible ? 1.0 : 0.0)
                }
                .onAppear {
                    withAnimation(.easeOut(duration: 0.3)) {
                        isVisible = true
                    }
                }
            case .recordHover:
                HStack(spacing: 4) {
                    RoundedRectangle(cornerRadius: 6)
                        .fill(Color.blue.opacity(0.2))
                        .stroke(Color.blue.opacity(0.3), lineWidth: 1)
                        .frame(width: 40, height: 24)
                        .scaleEffect(x: isVisible ? 1.0 : 0.0, y: 1.0, anchor: .trailing)
                        .opacity(isVisible ? 1.0 : 0.0)
                    Circle()
                        .fill(Color.red)
                        .frame(width: 20, height: 20)
                }
                .onAppear {
                    withAnimation(.easeOut(duration: 0.3)) {
                        isVisible = true
                    }
                }
            case .promptTap:
                HStack(spacing: 4) {
                    Circle()
                        .fill(Color.red)
                        .frame(width: 20, height: 20)
                    RoundedRectangle(cornerRadius: 6)
                        .fill(Color.blue.opacity(0.2))
                        .stroke(Color.blue.opacity(0.3), lineWidth: 1)
                        .frame(width: 280, height: 24)
                        .scaleEffect(x: isVisible ? 1.0 : 0.0, y: 1.0, anchor: .leading)
                        .opacity(isVisible ? 1.0 : 0.0)
                }
                .onAppear {
                    withAnimation(.easeOut(duration: 0.3)) {
                        isVisible = true
                    }
                }
            case .recordTap:
                VStack(spacing: 8) {
                    // Pause button at top
                    RoundedRectangle(cornerRadius: 4)
                        .fill(Color.red)
                        .frame(width: 32, height: 32)
                        .overlay(
                            HStack(spacing: 3) {
                                Rectangle()
                                    .fill(Color.white)
                                    .frame(width: 3, height: 12)
                                Rectangle()
                                    .fill(Color.white)
                                    .frame(width: 3, height: 12)
                            }
                        )
                        .scaleEffect(isVisible ? 1.0 : 0.0)
                        .opacity(isVisible ? 1.0 : 0.0)
                    
                    // Text with waveform
                    HStack(spacing: 8) {
                        Text("hello world")
                            .foregroundColor(.primary)
                            .font(.system(size: 14))
                        AudioWaveform()
                            .frame(width: 160, height: 24)
                        RoundedRectangle(cornerRadius: 4)
                            .fill(Color.gray.opacity(0.3))
                            .frame(width: 24, height: 24)
                    }
                    .scaleEffect(x: isVisible ? 1.0 : 0.0, y: 1.0, anchor: .leading)
                    .opacity(isVisible ? 1.0 : 0.0)
                    
                    // Red square record button at bottom
                    RoundedRectangle(cornerRadius: 4)
                        .fill(Color.red)
                        .frame(width: 24, height: 24)
                        .scaleEffect(isVisible ? 1.0 : 0.0)
                        .opacity(isVisible ? 1.0 : 0.0)
                }
                .onAppear {
                    withAnimation(.easeOut(duration: 0.3)) {
                        isVisible = true
                    }
                }
            case .promptEnter:
                RoundedRectangle(cornerRadius: 6)
                    .fill(Color.blue.opacity(0.1))
                    .stroke(Color.blue.opacity(0.3), lineWidth: 1)
                    .frame(width: 200, height: 120)
                    .overlay(
                        TextEditor(text: $inputText)
                            .padding(8)
                            .background(Color.clear)
                            .font(.system(size: 14))
                    )
                    .scaleEffect(isVisible ? 1.0 : 0.0, anchor: .topLeading)
                    .opacity(isVisible ? 1.0 : 0.0)
                    .onAppear {
                        withAnimation(.easeOut(duration: 0.3)) {
                            isVisible = true
                        }
                    }
            case .promptAnswer:
                VStack(alignment: .leading, spacing: 8) {
                    HStack(spacing: 4) {
                        Circle()
                            .fill(Color.red)
                            .frame(width: 20, height: 20)
                        RoundedRectangle(cornerRadius: 6)
                            .fill(Color.blue.opacity(0.2))
                            .stroke(Color.blue.opacity(0.3), lineWidth: 1)
                            .frame(width: 80, height: 24)
                            .scaleEffect(x: isVisible ? 1.0 : 0.0, y: 1.0, anchor: .leading)
                            .opacity(isVisible ? 1.0 : 0.0)
                    }
                    RoundedRectangle(cornerRadius: 6)
                        .fill(Color.blue.opacity(0.1))
                        .stroke(Color.blue.opacity(0.2), lineWidth: 1)
                        .frame(width: 280, height: 80)
                        .scaleEffect(isVisible ? 1.0 : 0.0, anchor: .topLeading)
                        .opacity(isVisible ? 1.0 : 0.0)
                }
                .onAppear {
                    withAnimation(.easeOut(duration: 0.3)) {
                        isVisible = true
                    }
                }
            case .promptHistory:
                VStack(alignment: .leading, spacing: 8) {
                    HStack(spacing: 4) {
                        Circle()
                            .fill(Color.red)
                            .frame(width: 20, height: 20)
                        RoundedRectangle(cornerRadius: 6)
                            .fill(Color.blue.opacity(0.2))
                            .stroke(Color.blue.opacity(0.3), lineWidth: 1)
                            .frame(width: 80, height: 24)
                            .scaleEffect(x: isVisible ? 1.0 : 0.0, y: 1.0, anchor: .leading)
                            .opacity(isVisible ? 1.0 : 0.0)
                    }
                    RoundedRectangle(cornerRadius: 6)
                        .fill(Color.gray.opacity(0.1))
                        .stroke(Color.gray.opacity(0.2), lineWidth: 1)
                        .frame(width: 280, height: 100)
                        .scaleEffect(isVisible ? 1.0 : 0.0, anchor: .topLeading)
                        .opacity(isVisible ? 1.0 : 0.0)
                }
                .onAppear {
                    withAnimation(.easeOut(duration: 0.3)) {
                        isVisible = true
                    }
                }
            case .promptLarge:
                HStack(spacing: 4) {
                    Circle()
                        .fill(Color.red)
                        .frame(width: 20, height: 20)
                    RoundedRectangle(cornerRadius: 6)
                        .fill(Color.blue.opacity(0.2))
                        .stroke(Color.blue.opacity(0.3), lineWidth: 1)
                        .frame(width: 320, height: 24)
                        .scaleEffect(x: isVisible ? 1.0 : 0.0, y: 1.0, anchor: .leading)
                        .opacity(isVisible ? 1.0 : 0.0)
                }
                .onAppear {
                    withAnimation(.easeOut(duration: 0.3)) {
                        isVisible = true
                    }
                }
            case .recordLarge:
                HStack(spacing: 8) {
                    RoundedRectangle(cornerRadius: 4)
                        .fill(Color.gray.opacity(0.3))
                        .frame(width: 24, height: 24)
                        .scaleEffect(isVisible ? 1.0 : 0.0)
                        .opacity(isVisible ? 1.0 : 0.0)
                    AudioWaveform()
                        .frame(width: 200, height: 24)
                        .scaleEffect(x: isVisible ? 1.0 : 0.0, y: 1.0, anchor: .trailing)
                        .opacity(isVisible ? 1.0 : 0.0)
                    Text("hello world")
                        .foregroundColor(.primary)
                        .font(.system(size: 14))
                }
                .onAppear {
                    withAnimation(.easeOut(duration: 0.3)) {
                        isVisible = true
                    }
                }
            default:
                TextField("Enter text...", text: $inputText, axis: .vertical)
                    .textFieldStyle(.plain)
                    .padding(EdgeInsets(top: 8, leading: 12, bottom: 8, trailing: 12))
                    .background(
                        RoundedRectangle(cornerRadius: 8)
                            .fill(Color.white.opacity(0.3))
                            .background(.ultraThinMaterial, in: RoundedRectangle(cornerRadius: 8))
                            .blur(radius: 10)
                    )
                    .shadow(color: Color.black.opacity(0.1), radius: 0, x: 0, y: 2)
                    .frame(
                        width: isHovered ? 400 : 100,
                        height: 32
                    )
                    .scaleEffect(x: isVisible ? 1.0 : 0.0, y: 1.0, anchor: .leading)
                    .opacity(isVisible ? 1.0 : 0.0)
                    .onHover { hovering in
                        withAnimation(.bouncy(duration: 0.4)) {
                            isHovered = hovering
                        }
                    }
                    .onAppear {
                        withAnimation(.bouncy(duration: 0.6)) {
                            isVisible = true
                        }
                    }
            }
        }
    }
}

// MARK: - Cursor State Management

class InteractiveCursorManager: ObservableObject {
    @Published var isMenuVisible: Bool = false
    @Published var cursorPosition: CGPoint = .zero
    @Published var isHovered: Bool = false
    
    func showMenu(at position: CGPoint) {
        cursorPosition = position
        isMenuVisible = true
    }
    
    func hideMenu() {
        isMenuVisible = false
    }
}

/// A component that manages a custom interactive caret and its associated menu in a text view.
class SmartCaret {
    weak var textView: CustomInlineNSTextView?
    
    /// Core animation layer representing the caret.
    private var interactiveCursorLayer: CALayer?
    private var cursorTrackingArea: NSTrackingArea?
    private var menuHostingView: NSHostingView<CursorMenu>?
    private var cursorManager = InteractiveCursorManager()
    private var cursorUpdateTimer: Timer?
    private var menuState: CursorMenuState = .inactive
    
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
        layer.backgroundColor = NSColor.systemRed.cgColor
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
        let height: CGFloat = 24.0 // Fixed height to match the design
        let frame = CGRect(x: viewRect.minX,
                           y: viewRect.minY + (viewRect.height - height) / 2, // Center vertically
                           width: width,
                           height: height)
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
            layer.backgroundColor = NSColor.systemRed.cgColor
            layer.transform = CATransform3DMakeScale(1.5, 1.0, 1.0)
        } else {
            layer.backgroundColor = NSColor.systemRed.cgColor
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
                // Определяем, с какой стороны от курсора был клик
                let isOnLeftSide = local.x < layer.frame.midX
                showMenu(event: event, isPromptSide: !isOnLeftSide)
            }
        } else {
            hideMenu()
        }
    }
    
    /// Show the SwiftUI menu at the caret location.
    private func showMenu(event: NSEvent, isPromptSide: Bool = true) {
        guard let view = textView, let caretLayer = interactiveCursorLayer else { return }
        
        hideMenu()
        
        // Determine state: promptTap if prompt side, else recordTap
        let selectedState: CursorMenuState = isPromptSide ? .promptTap : .recordTap
        let menu = CursorMenu(state: selectedState)
        let host = NSHostingView(rootView: menu)
        host.translatesAutoresizingMaskIntoConstraints = false
        
        view.superview?.addSubview(host)
        
        let caretFrameInView = caretLayer.frame
        let caretRectInWindow = view.convert(caretFrameInView, to: nil)
        let hostTargetSuperview = view.superview ?? view
        let caretRectInSuperview = hostTargetSuperview.convert(caretRectInWindow, from: nil)

        // Adjust sizes based on state
        let menuHeight: CGFloat = selectedState == .recordTap ? 100 : 24
        let menuWidth: CGFloat = selectedState == .promptTap ? 320 : 280
        
        let menuX: CGFloat
        if isPromptSide {
            // Prompt справа от курсора
            menuX = caretRectInSuperview.maxX + 8
        } else {
            // Record слева от курсора
            menuX = caretRectInSuperview.minX - menuWidth - 8
        }
        
        let menuY = caretRectInSuperview.midY - (menuHeight / 2)
        
        // Just set the frame, let SwiftUI handle all animations
        host.frame = CGRect(x: menuX, y: menuY, width: menuWidth, height: menuHeight)
        
        menuHostingView = host
        cursorManager.showMenu(at: layerPosition())
    }
    
    /// Hide and remove the menu.
    private func hideMenu() {
        guard let host = menuHostingView else { return }
        
        // Simply remove without NSView animations that cause conflicts
        host.removeFromSuperview()
        
        cursorManager.hideMenu()
        menuHostingView = nil
    }
    
    /// Show hover menu without an event, using current caret position.
    private func showHoverMenu(isPromptSide: Bool = true) {
        guard let view = textView, let caretLayer = interactiveCursorLayer else { return }
        hideMenu()
        menuState = isPromptSide ? .promptHover : .recordHover
        let menu = CursorMenu(state: menuState)
        let host = NSHostingView(rootView: menu)
        host.translatesAutoresizingMaskIntoConstraints = false
        view.superview?.addSubview(host)
        let caretFrameInView = caretLayer.frame
        let caretRectInWindow = view.convert(caretFrameInView, to: nil)
        let hostTargetSuperview = view.superview ?? view
        let caretRectInSuperview = hostTargetSuperview.convert(caretRectInWindow, from: nil)
        let menuHeight: CGFloat = 24
        let menuWidth: CGFloat = isPromptSide ? 120 : 80
        
        let menuX: CGFloat
        if isPromptSide {
            // Prompt поле справа от курсора
            menuX = caretRectInSuperview.maxX + 8
        } else {
            // Record поле слева от курсора
            menuX = caretRectInSuperview.minX - menuWidth - 8
        }
        
        let menuY = caretRectInSuperview.midY - (menuHeight / 2)
        host.frame = CGRect(x: menuX, y: menuY, width: menuWidth, height: menuHeight)
        menuHostingView = host
        cursorManager.showMenu(at: layerPosition())
    }

    func handleMouseMoved(event: NSEvent) {
        guard let view = textView, let layer = interactiveCursorLayer else { return }
        let windowPoint = event.locationInWindow
        // If menu is visible, check if pointer is over menu or buffer zone between caret and menu
        if let host = menuHostingView {
            // Over menu itself
            let pointInHost = host.convert(windowPoint, from: nil)
            if host.bounds.contains(pointInHost) {
                return
            }
            // Buffer zone between caret and menu
            let caretRectInView = layer.frame
            let caretRectInWindow = view.convert(caretRectInView, to: nil)
            let hostSuperview = view.superview ?? view
            let caretRectInSuper = hostSuperview.convert(caretRectInWindow, from: nil)
            let menuFrame = host.frame
            let bufferRect = caretRectInSuper.union(menuFrame).insetBy(dx: -8, dy: -8)
            let pointInSuper = hostSuperview.convert(windowPoint, from: nil)
            if bufferRect.contains(pointInSuper) {
                return
            }
        }
        // Convert event point into text view coordinates
        let local = view.convert(windowPoint, from: nil)
        let hoverArea = layer.frame.insetBy(dx: -10, dy: -5)
        let wasHovered = cursorManager.isHovered
        cursorManager.isHovered = hoverArea.contains(local)
        if wasHovered != cursorManager.isHovered {
            updateAppearance()
            if cursorManager.isHovered {
                // Determine side relative to cursor
                let isOnLeftSide = local.x < layer.frame.midX
                showHoverMenu(isPromptSide: !isOnLeftSide)
            } else {
                hideMenu()
            }
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
        interactiveCursorLayer?.removeFromSuperlayer()
        hideMenu()
    }
}

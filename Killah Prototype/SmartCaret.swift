import SwiftUI
import AppKit
import QuartzCore

// MARK: - Interactive Cursor SwiftUI Components

enum CursorMenuType {
    case promptInput
    case audioRecording
    case textFormatting
    case aiActions
}

struct CursorMenu: View {
    let menuType: CursorMenuType
    let onDismiss: () -> Void
    @State private var promptText: String = ""
    @State private var isRecording: Bool = false
    
    var body: some View {
        VStack(spacing: 12) {
            switch menuType {
            case .promptInput:
                promptInputMenu
            case .audioRecording:
                audioRecordingMenu
            case .textFormatting:
                textFormattingMenu
            case .aiActions:
                aiActionsMenu
            }
        }
        .padding(16)
        .background(.regularMaterial, in: RoundedRectangle(cornerRadius: 12))
        .shadow(color: .black.opacity(0.1), radius: 8, x: 0, y: 4)
    }
    
    private var promptInputMenu: some View {
        VStack(spacing: 8) {
            Text("AI Prompt")
                .font(.headline)
                .foregroundColor(.primary)
            
            TextField("Enter your prompt...", text: $promptText)
                .textFieldStyle(.roundedBorder)
                .frame(width: 250)
            
            HStack(spacing: 8) {
                Button("Cancel") {
                    onDismiss()
                }
                .buttonStyle(.bordered)
                
                Button("Generate") {
                    // Handle prompt generation
                    print("Generating with prompt: \(promptText)")
                    onDismiss()
                }
                .buttonStyle(.borderedProminent)
                .disabled(promptText.isEmpty)
            }
        }
    }
    
    private var audioRecordingMenu: some View {
        VStack(spacing: 8) {
            Text("Voice Input")
                .font(.headline)
                .foregroundColor(.primary)
            
            Button(action: {
                isRecording.toggle()
            }) {
                HStack {
                    Image(systemName: isRecording ? "stop.circle.fill" : "mic.circle.fill")
                        .font(.title)
                        .foregroundColor(isRecording ? .red : .blue)
                    
                    Text(isRecording ? "Stop Recording" : "Start Recording")
                        .font(.body)
                }
            }
            .buttonStyle(.bordered)
            
            if isRecording {
                HStack {
                    Circle()
                        .fill(.red)
                        .frame(width: 8, height: 8)
                        .opacity(0.8)
                    
                    Text("Recording...")
                        .font(.caption)
                        .foregroundColor(.secondary)
                }
            }
            
            Button("Close") {
                onDismiss()
            }
            .buttonStyle(.bordered)
        }
    }
    
    private var textFormattingMenu: some View {
        VStack(spacing: 8) {
            Text("Text Formatting")
                .font(.headline)
                .foregroundColor(.primary)
            
            LazyVGrid(columns: Array(repeating: GridItem(.flexible()), count: 3), spacing: 8) {
                FormatButton(icon: "bold", title: "Bold") { /* Handle bold */ }
                FormatButton(icon: "italic", title: "Italic") { /* Handle italic */ }
                FormatButton(icon: "underline", title: "Underline") { /* Handle underline */ }
                FormatButton(icon: "list.bullet", title: "Bullets") { /* Handle bullets */ }
                FormatButton(icon: "list.number", title: "Numbers") { /* Handle numbers */ }
                FormatButton(icon: "highlighter", title: "Highlight") { /* Handle highlight */ }
            }
            
            Button("Close") {
                onDismiss()
            }
            .buttonStyle(.bordered)
        }
    }
    
    private var aiActionsMenu: some View {
        VStack(spacing: 8) {
            Text("AI Actions")
                .font(.headline)
                .foregroundColor(.primary)
            
            VStack(spacing: 4) {
                ActionButton(title: "Continue Writing", icon: "pencil.line") {
                    print("Continue writing")
                    onDismiss()
                }
                ActionButton(title: "Improve Text", icon: "wand.and.stars") {
                    print("Improve text")
                    onDismiss()
                }
                ActionButton(title: "Summarize", icon: "text.alignleft") {
                    print("Summarize")
                    onDismiss()
                }
                ActionButton(title: "Translate", icon: "globe") {
                    print("Translate")
                    onDismiss()
                }
            }
            
            Button("Close") {
                onDismiss()
            }
            .buttonStyle(.bordered)
        }
    }
}

struct FormatButton: View {
    let icon: String
    let title: String
    let action: () -> Void
    
    var body: some View {
        Button(action: action) {
            VStack(spacing: 4) {
                Image(systemName: icon)
                    .font(.caption)
                Text(title)
                    .font(.caption2)
            }
            .frame(width: 60, height: 40)
        }
        .buttonStyle(.bordered)
    }
}

struct ActionButton: View {
    let title: String
    let icon: String
    let action: () -> Void
    
    var body: some View {
        Button(action: action) {
            HStack {
                Image(systemName: icon)
                    .font(.body)
                Text(title)
                    .font(.body)
                Spacer()
            }
            .frame(maxWidth: .infinity)
            .padding(.horizontal, 8)
            .padding(.vertical, 4)
        }
        .buttonStyle(.bordered)
    }
}

// MARK: - Cursor State Management

class InteractiveCursorManager: ObservableObject {
    @Published var isMenuVisible: Bool = false
    @Published var currentMenuType: CursorMenuType = .promptInput
    @Published var cursorPosition: CGPoint = .zero
    @Published var isHovered: Bool = false
    
    func showMenu(type: CursorMenuType, at position: CGPoint) {
        currentMenuType = type
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

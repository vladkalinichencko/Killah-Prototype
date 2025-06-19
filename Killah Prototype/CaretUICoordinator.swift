import SwiftUI
import Combine
import AppKit

class NonResponderHostingView<Content: View>: NSHostingView<Content> {
    override var acceptsFirstResponder: Bool { false }
}

class CaretUICoordinator: ObservableObject {
    @Published var caretPosition: CGPoint = .zero
    @Published var caretSize: CGSize = CGSize(width: 2, height: 20)
    @Published var isExpanded: Bool = false
    
    @Published var isCaretHovered: Bool = false
    @Published var isCaretPressed: Bool = false
    
    @Published var promptText: String = ""
    @Published var isPromptFieldHovered: Bool = false
    
    @Published var isRecording: Bool = false
    @Published var isPaused: Bool = false
    @Published var transcribedText: String = ""
    @Published var audioLevel: Float = 0.0

    private var audioEngine: AudioEngine
    private let fontManager = FontManager.shared
    private var cancellables = Set<AnyCancellable>()
    
    private let basePromptFieldWidth: CGFloat = 150
    private let expandedPromptFieldWidth: CGFloat = 300
    
    var editorFontSize: CGFloat { fontManager.defaultEditorFontSize }
    var fixedMenuItemSize: CGFloat { fontManager.menuItemSize }
    var fixedPromptFieldHeight: CGFloat { fontManager.promptFieldHeight }
    var fixedPromptFieldFontSize: CGFloat { fontManager.promptFieldFontSize }

    var caretButtonPadding: CGFloat = 24

    init(audioEngine: AudioEngine) {
        self.audioEngine = audioEngine

        // Bind AudioEngine properties to CaretUICoordinator properties
        audioEngine.$isRecording
            .receive(on: DispatchQueue.main)
            .sink { [weak self] isRecording in
                self?.isRecording = isRecording
            }
            .store(in: &cancellables)

        audioEngine.$isPaused
            .receive(on: DispatchQueue.main)
            .sink { [weak self] isPaused in
                self?.isPaused = isPaused
            }
            .store(in: &cancellables)

        audioEngine.$transcribedText
            .receive(on: DispatchQueue.main)
            .sink { [weak self] text in
                self?.transcribedText = text
            }
            .store(in: &cancellables)

        audioEngine.$audioLevel
            .receive(on: DispatchQueue.main)
            .sink { [weak self] level in
                self?.audioLevel = level
            }
            .store(in: &cancellables)
    }

    func updateCaretPosition(for textView: NSTextView, at charIndex: Int? = nil) {
        let currentInsertionPoint: Int
        if let charIndex = charIndex {
            currentInsertionPoint = charIndex
        } else {
            currentInsertionPoint = textView.selectedRange().location
        }

        let textLength = (textView.string as NSString).length
        let insertionPoint = max(0, min(currentInsertionPoint, textLength))

        var finalCaretX: CGFloat?
        var finalCaretY: CGFloat?
        var finalCaretHeight: CGFloat?

        let charNSRange = NSRange(location: insertionPoint, length: 0)
        let screenRect = textView.firstRect(forCharacterRange: charNSRange, actualRange: nil)
        if screenRect.height > 0, let window = textView.window {
            let windowRect = window.convertFromScreen(screenRect)
            let localOrigin = textView.convert(windowRect.origin, from: nil)
            finalCaretX = localOrigin.x
            finalCaretY = localOrigin.y
            finalCaretHeight = screenRect.height
        }

        if let x = finalCaretX, let y = finalCaretY, let height = finalCaretHeight,
           !x.isNaN, !x.isInfinite, !y.isNaN, !y.isInfinite, !height.isNaN, !height.isInfinite, height > 0 {
            DispatchQueue.main.async {
                self.caretPosition = CGPoint(x: x, y: y - height / 2)
                self.caretSize = CGSize(width: 2, height: height)
            }
        } else {
            setDefaultCaretPosition(for: textView)
        }
    }

    private func setDefaultCaretPosition(for textView: NSTextView) {
        let defaultHeight = textView.font?.pointSize ?? 16
        DispatchQueue.main.async {
            self.caretPosition = CGPoint(x: textView.textContainerOrigin.x, y: textView.textContainerOrigin.y + defaultHeight / 2)
            self.caretSize = CGSize(width: 2, height: defaultHeight)
        }
    }
    
    func toggleExpanded() {
        withAnimation(.spring(response: 0.3, dampingFraction: 0.8, blendDuration: 0.1)) {
            isExpanded.toggle()
        }
    }
    
    func setExpanded(_ expanded: Bool) {
        withAnimation(.spring(response: 0.3, dampingFraction: 0.8, blendDuration: 0.1)) {
            isExpanded = expanded
        }
    }
    
    func collapseUI() {
        setExpanded(false)
    }
    
    func startRecording() {
        audioEngine.startRecording()
    }
    
    func stopRecording() {
        audioEngine.stopRecording()
    }
    
    func togglePause() {
        audioEngine.togglePause()
    }
    
    var caretColor: Color {
        if isRecording {
            return Color.red.opacity(0.7)
        } else if isCaretHovered {
            return Color.red
        } else {
            return Color.primary
        }
    }
    
    var caretWidth: CGFloat {
        if isCaretPressed {
            return caretSize.width * 1.2
        } else if isCaretHovered {
            return caretSize.width * 1.4
        } else {
            return caretSize.width
        }
    }
    
    var caretHeight: CGFloat {
        return caretSize.height
    }
    
    var shadowColor: Color {
        if isCaretPressed {
            return Color.red.opacity(0.4)
        } else if isCaretHovered {
            return Color.red.opacity(0.7)
        } else {
            return Color.clear
        }
    }
    
    var shadowRadius: CGFloat {
        if isCaretPressed {
            return 5
        } else if isCaretHovered {
            return 10
        } else {
            return 0
        }
    }
    
    func caretOverlayFrame() -> CGRect {
        return CGRect(
            x: caretPosition.x - 10,
            y: caretPosition.y - caretSize.height / 2 - 10,
            width: 20,
            height: caretSize.height + 20
        )
    }
    
    func recordButtonFrame() -> CGRect {
        return CGRect(
            x: caretPosition.x - caretButtonPadding * 2,
            y: caretPosition.y - fixedMenuItemSize / 2,
            width: fixedMenuItemSize,
            height: fixedMenuItemSize
        )
    }
    
    func promptFieldFrame() -> CGRect {
        let promptWidth = promptFieldWidth
        let promptHeight = calculatePromptFieldHeight()
        let baseX = caretPosition.x + caretButtonPadding
        let compensatedX = isPromptFieldExpanded ? baseX - promptFieldWidthOffset : baseX

        return CGRect(
            x: compensatedX,
            y: caretPosition.y - promptHeight / 2,
            width: promptWidth,
            height: promptHeight
        )
    }
    
    func calculatePromptFieldHeight() -> CGFloat {
        let baseHeight = fixedPromptFieldHeight
        let maxHeight = baseHeight * 3
        let padding: CGFloat = 12

        let textStorage = NSTextStorage(string: promptText)
        let textContainer = NSTextContainer(size: CGSize(width: promptFieldWidth - 24, height: CGFloat.greatestFiniteMagnitude))
        let layoutManager = NSLayoutManager()
        layoutManager.addTextContainer(textContainer)
        textStorage.addLayoutManager(layoutManager)

        let attributes: [NSAttributedString.Key: Any] = [
            .font: NSFont.systemFont(ofSize: fixedPromptFieldFontSize)
        ]
        textStorage.addAttributes(attributes, range: NSRange(location: 0, length: textStorage.length))
        
        layoutManager.ensureLayout(for: textContainer)
        let usedRect = layoutManager.usedRect(for: textContainer)
        
        let calculatedHeight = ceil(usedRect.height) + padding
        
        return min(max(baseHeight, calculatedHeight), maxHeight)
    }

    var isPromptFieldExpanded: Bool {
        isPromptFieldHovered || !promptText.isEmpty
    }
    
    var promptFieldWidth: CGFloat {
        return CGFloat(isPromptFieldExpanded ? expandedPromptFieldWidth : basePromptFieldWidth)
    }
    
    var promptFieldWidthOffset: CGFloat {
        isPromptFieldExpanded ? (expandedPromptFieldWidth - basePromptFieldWidth) / 2 : 0
    }
    
    func createCaretOverlay() -> NonResponderHostingView<SmartCaretView> {
        let overlay = NonResponderHostingView(rootView: SmartCaretView(coordinator: self))
        overlay.translatesAutoresizingMaskIntoConstraints = true
        return overlay
    }
    
    func createRecordButtonOverlay() -> NonResponderHostingView<CaretRecordButton> {
        let overlay = NonResponderHostingView(rootView: CaretRecordButton(coordinator: self))
        overlay.translatesAutoresizingMaskIntoConstraints = true
        return overlay
    }
    
    func createPromptFieldOverlay() -> NonResponderHostingView<CaretPromptField> {
        let view = CaretPromptField(coordinator: self)
        let hostingView = NonResponderHostingView(rootView: view)
        hostingView.translatesAutoresizingMaskIntoConstraints = true
        return hostingView
    }
    
    func pauseButtonFrame() -> CGRect {
        return CGRect(
            x: caretPosition.x - fixedMenuItemSize / 2,
            y: caretPosition.y - fixedMenuItemSize / 2 - caretButtonPadding,
            width: fixedMenuItemSize,
            height: fixedMenuItemSize
        )
    }
    
    func stopButtonFrame() -> CGRect {
        let x = caretPosition.x - (fixedMenuItemSize) / 2
        let y = caretPosition.y + caretButtonPadding
        return CGRect(x: x, y: y, width: fixedMenuItemSize, height: fixedMenuItemSize)
    }
    
    func audioWaveformFrame() -> CGRect {
        let width: CGFloat = 150
        let x = caretPosition.x + caretButtonPadding
        let y = caretPosition.y - (fixedMenuItemSize + 12) / 2
        return CGRect(x: x, y: y, width: width, height: fixedMenuItemSize + 12)
    }
    
    func transcriptionFrame() -> CGRect {
        let width: CGFloat = 150
        let x = caretPosition.x - width - caretButtonPadding
        let y = caretPosition.y - (fixedMenuItemSize + 12) / 2
        return CGRect(x: x, y: y, width: width, height: fixedMenuItemSize + 12)
    }
    
    func createPauseButtonOverlay() -> NonResponderHostingView<CaretPauseButton> {
        let view = CaretPauseButton(coordinator: self)
        return NonResponderHostingView(rootView: view)
    }
    
    func createStopButtonOverlay() -> NonResponderHostingView<CaretStopButton> {
        let view = CaretStopButton(coordinator: self)
        return NonResponderHostingView(rootView: view)
    }
    
    func createAudioWaveformOverlay() -> NonResponderHostingView<AudioWaveformView> {
        let view = AudioWaveformView(coordinator: self)
        return NonResponderHostingView(rootView: view)
    }
    
    func createTranscriptionOverlay() -> NonResponderHostingView<TranscriptionView> {
        let view = TranscriptionView(coordinator: self)
        return NonResponderHostingView(rootView: view)
    }
}

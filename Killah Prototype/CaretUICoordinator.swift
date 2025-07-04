import SwiftUI
import Combine
import AppKit

class NonResponderHostingView<Content: View>: NSHostingView<Content> {
    override var acceptsFirstResponder: Bool { true }
}

class CaretUICoordinator: ObservableObject {
    // Триггер для caret-эффекта (анимации)
    @Published var triggerCaretEffect: Bool = false
    @Published var triggerBounceRight: Bool = false
    @Published var triggerBounceLeft: Bool = false
    @Published var caretPositionInWindow: CGPoint = .zero
    @Published var caretSize: CGSize = CGSize(width: 2, height: 20)
    
    // Basic caret state for coordinate calculations
    @Published var isExpanded: Bool = false
    
    // Audio engine state (read-only from coordinator perspective)
    @Published var isRecording: Bool = false
    @Published var isPaused: Bool = false
    @Published var transcribedText: String = ""
    @Published var audioLevel: Float = 0.0
    @Published var isProcessingAudio: Bool = false

    // User input
    @Published var promptText: String = ""
    
    // Computed property - overlay should show ONLY during recording, not during processing
    var shouldShowOverlay: Bool {
        return isRecording
    }

    private var audioEngine: AudioEngine
    private var llmEngine: LLMEngine
    private let fontManager = FontManager.shared
    private var cancellables = Set<AnyCancellable>()
    
    // Font and size properties from FontManager
    var editorFontSize: CGFloat { fontManager.defaultEditorFontSize }
    var menuItemSize: CGFloat { fontManager.menuItemSize }
    var promptFieldHeight: CGFloat { fontManager.promptFieldHeight }
    var promptFieldFontSize: CGFloat { fontManager.promptFieldFontSize }

    // Basic layout constants
    let basePromptFieldWidth: CGFloat = 150
    let expandedPromptFieldWidth: CGFloat = 300
    let caretButtonPadding: CGFloat = 24

    init(llmEngine: LLMEngine, audioEngine: AudioEngine) {
        self.llmEngine = llmEngine
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

        audioEngine.$isProcessingAudio
            .receive(on: DispatchQueue.main)
            .sink { [weak self] isProcessing in
                self?.isProcessingAudio = isProcessing
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

        var finalCaretPos: CGPoint?
        var finalCaretHeight: CGFloat?

        if let layoutManager = textView.layoutManager, let textContainer = textView.textContainer {
            let glyphRange = layoutManager.glyphRange(forCharacterRange: NSRange(location: insertionPoint, length: 0), actualCharacterRange: nil)
            var localRect = layoutManager.boundingRect(forGlyphRange: glyphRange, in: textContainer)

            // The rect is relative to the text container's origin. We need to offset it by that origin
            // to get it into the textView's coordinate space.
            let containerOrigin = textView.textContainerOrigin
            localRect.origin.x += containerOrigin.x
            localRect.origin.y += containerOrigin.y

            if localRect.height > 0, let window = textView.window, let contentView = window.contentView {
                // Конвертируем прямоугольник из локальных координат textView в координаты contentView окна
                let rectInContentView = textView.convert(localRect, to: contentView)

                let centerX = rectInContentView.origin.x + (rectInContentView.width / 2)
                let verticalFineTune: CGFloat = 2
                let centerY = rectInContentView.origin.y - rectInContentView.height + verticalFineTune

                finalCaretPos = CGPoint(x: centerX, y: centerY)
                finalCaretHeight = rectInContentView.height
            }
        }

        if let pos = finalCaretPos, let height = finalCaretHeight,
           !pos.x.isNaN, !pos.x.isInfinite, !pos.y.isNaN, !pos.y.isInfinite, height > 0 {
            DispatchQueue.main.async {
                // Проверяем, изменилась ли позиция, чтобы избежать лишних обновлений
                if self.caretPositionInWindow != pos || self.caretSize.height != height {
                    self.caretPositionInWindow = pos
                    self.caretSize = CGSize(width: 2, height: height)
                    // If the caret moves, collapse the UI.
                    if self.isExpanded {
                        self.collapseUI()
                    }
                }
            }
        }
        // Больше не сбрасываем позицию на дефолтную при ошибке, чтобы избежать застревания
    }

    // Simple state toggle without animation (views handle their own animations)
    func toggleExpanded() {
        isExpanded.toggle()
    }
    
    func setExpanded(_ expanded: Bool) {
        isExpanded = expanded
    }
    
    func collapseUI() {
        setExpanded(false)
    }
    
    // Audio control delegation
    func startRecording() {
        audioEngine.startRecording()
    }
    
    func stopRecording() {
        audioEngine.stopRecording()
    }
    
    func togglePause() {
        audioEngine.togglePause()
    }
    
    // Helper function for prompt field height calculation
    func calculatePromptFieldHeight() -> CGFloat {
        let font = NSFont.systemFont(ofSize: promptFieldFontSize)
        // Используем expandedPromptFieldWidth для расчёта высоты (или basePromptFieldWidth, если не расширено)
        let width = expandedPromptFieldWidth - 24
        let text = promptText.isEmpty ? " " : promptText
        let nsText = text as NSString
        let boundingRect = nsText.boundingRect(
            with: CGSize(width: width, height: .greatestFiniteMagnitude),
            options: [.usesLineFragmentOrigin, .usesFontLeading],
            attributes: [.font: font],
            context: nil
        )
        // Корректная высота строки для NSFont
        let lineHeight = font.ascender - font.descender + font.leading
        let numberOfLines = max(1, Int(ceil(boundingRect.height / lineHeight)))
        let minHeight = promptFieldHeight
        let maxHeight = minHeight * 3
        let totalHeight = CGFloat(numberOfLines) * lineHeight + 12 // 12 — паддинг
        return max(minHeight, min(totalHeight, maxHeight))
    }

    /// Вызвать caret-эффект (анимацию)
    func triggerCaretGenerationEffect() {
        triggerCaretEffect = true
    }
}

private extension String {
    func height(withConstrainedWidth width: CGFloat, font: NSFont) -> CGFloat {
        let constraintRect = CGSize(width: width, height: .greatestFiniteMagnitude)
        let boundingBox = self.boundingRect(with: constraintRect, options: .usesLineFragmentOrigin, attributes: [.font: font], context: nil)
        return ceil(boundingBox.height)
    }

    func width(withConstrainedHeight height: CGFloat, font: NSFont) -> CGFloat {
        let constraintRect = CGSize(width: .greatestFiniteMagnitude, height: height)
        let boundingBox = self.boundingRect(with: constraintRect, options: .usesLineFragmentOrigin, attributes: [.font: font], context: nil)
        return ceil(boundingBox.width)
    }
}

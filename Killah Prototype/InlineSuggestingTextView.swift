import Combine
import SwiftUI
import AppKit
import QuartzCore

struct InlineSuggestingTextView: NSViewRepresentable {
    @Binding var text: String
    @ObservedObject var llmEngine: LLMEngine
    @Binding var debouncer: Debouncer
    @Binding var formattingDelegate: TextFormattingDelegate?
    
    private let fontManager = FontManager.shared

    func makeCoordinator() -> Coordinator {
        Coordinator(self, llmEngine: llmEngine)
    }

    func makeNSView(context: Context) -> NSScrollView {
        let textView = CustomInlineNSTextView(frame: .zero)
        
        textView.delegate = context.coordinator
        textView.font = fontManager.defaultEditorFont()
        textView.isEditable = true
        textView.isSelectable = true
        textView.allowsUndo = true
        textView.textContainerInset = CGSize(width: 10, height: 80)
        
        textView.drawsBackground = false 
        textView.backgroundColor = .clear

        textView.llmInteractionDelegate = context.coordinator
        context.coordinator.managedTextView = textView

        setupTextViewContent(textView, context: context)
        
        let scrollView = createScrollView(with: textView)
        
        setupCustomCaret(textView, context: context)
        DispatchQueue.main.async {
            textView.window?.makeFirstResponder(textView)
            context.coordinator.caretCoordinator?.updateCaretPosition(for: textView)
        }
        
        return scrollView
    }
    
    private func setupTextViewContent(_ textView: CustomInlineNSTextView, context: Context) {
        if !text.isEmpty {
            textView.string = text
        }
        context.coordinator.currentCommittedText = text
    }
    
    private func createScrollView(with textView: CustomInlineNSTextView) -> NSScrollView {
        let scrollView = NSScrollView()
        scrollView.hasVerticalScroller = true
        scrollView.borderType = .noBorder
        scrollView.drawsBackground = false
        scrollView.backgroundColor = .clear
        
        let clipView = scrollView.contentView
        clipView.drawsBackground = false
        clipView.backgroundColor = .clear
        
        scrollView.contentInsets = NSEdgeInsets(top: 150, left: 0, bottom: 0, right: 0)
        scrollView.scrollerInsets = NSEdgeInsets(top: 0, left: 0, bottom: 0, right: 0)
        scrollView.documentView = textView
        return scrollView
    }
    
    private func setupCustomCaret(_ textView: CustomInlineNSTextView, context: Context) {
        let caretCoordinator = CaretUICoordinator()
        context.coordinator.caretCoordinator = caretCoordinator
        
        let caretOverlay = caretCoordinator.createCaretOverlay()
        let menuOverlay = caretCoordinator.createRecordButtonOverlay()
        let promptOverlay = caretCoordinator.createPromptFieldOverlay()
        
        textView.addSubview(caretOverlay)
        textView.addSubview(menuOverlay)
        textView.addSubview(promptOverlay)

        caretOverlay.frame = caretCoordinator.caretOverlayFrame()
        menuOverlay.frame = caretCoordinator.recordButtonFrame()
        promptOverlay.frame = caretCoordinator.promptFieldFrame()
        
        menuOverlay.isHidden = !caretCoordinator.isExpanded
        promptOverlay.isHidden = !caretCoordinator.isExpanded
        
        DispatchQueue.main.async {
            caretCoordinator.updateCaretPosition(for: textView)
        }

        let caretCancellable = caretCoordinator.$caretPosition.combineLatest(caretCoordinator.$caretSize).sink { _, _ in
            NSAnimationContext.runAnimationGroup { context in
                context.duration = 0.1
                context.timingFunction = CAMediaTimingFunction(name: .easeInEaseOut)
                caretOverlay.animator().frame = caretCoordinator.caretOverlayFrame()
            }
        }
        
        let menuCancellable = caretCoordinator.$caretPosition
            .combineLatest(caretCoordinator.$caretSize, caretCoordinator.$isExpanded)
            .sink { _, _, isExpanded in
                NSAnimationContext.runAnimationGroup { context in
                    context.duration = 0.15
                    context.timingFunction = CAMediaTimingFunction(name: .easeInEaseOut)
                    menuOverlay.animator().frame = caretCoordinator.recordButtonFrame()
                    promptOverlay.animator().frame = caretCoordinator.promptFieldFrame()
                }
                menuOverlay.isHidden = !isExpanded
                promptOverlay.isHidden = !isExpanded
            }
        
        if context.coordinator.caretCancellables == nil {
            context.coordinator.caretCancellables = []
        }
        context.coordinator.caretCancellables?.append(caretCancellable)
        context.coordinator.caretCancellables?.append(menuCancellable)
    }
    

    func updateNSView(_ nsView: NSScrollView, context: Context) {
        guard let textView = nsView.documentView as? CustomInlineNSTextView else { return }
        
        if textView.committedText() != text && !context.coordinator.isInternallyUpdatingTextBinding {
            textView.clearGhostText()
            llmEngine.abortCurrentSuggestion()
            textView.string = text
            textView.selectedRange = NSRange(location: text.utf16.count, length: 0)
            DispatchQueue.main.async {
                context.coordinator.caretCoordinator?.updateCaretPosition(for: textView)
            }
            context.coordinator.currentCommittedText = text
        }
        
        if textView.isEditable != true {
            textView.isEditable = true
        }
    }

    
class Coordinator: NSObject, NSTextViewDelegate {
        var caretCancellables: [AnyCancellable]? = nil
        var fontCancellable: AnyCancellable?
        var caretCoordinator: CaretUICoordinator?
        var parent: InlineSuggestingTextView
        var llmEngine: LLMEngine
        weak var managedTextView: CustomInlineNSTextView?

        var currentCommittedText: String = ""
        var isInternallyUpdatingTextBinding: Bool = false
        var isProcessingAcceptOrDismiss: Bool = false

        init(_ parent: InlineSuggestingTextView, llmEngine: LLMEngine) {
            self.parent = parent
            self.llmEngine = llmEngine
            self.currentCommittedText = parent.text
            super.init()
            
            fontCancellable = FontManager.shared.$defaultEditorFontSize.sink { [weak self] newSize in
                DispatchQueue.main.async {
                    self?.updateDefaultFont()
                }
            }
        }
        
        deinit {
            fontCancellable?.cancel()
            caretCancellables?.forEach { $0.cancel() }
        }
        
        private func updateDefaultFont() {
            guard let textView = managedTextView else { return }
            textView.font = FontManager.shared.defaultEditorFont()
        }

        func textDidChange(_ notification: Notification) {
            guard let textView = notification.object as? CustomInlineNSTextView else { return }
            caretCoordinator?.updateCaretPosition(for: textView)
            if isProcessingAcceptOrDismiss { return }
            handleTextChange(for: textView)
        }
        
        func textView(_ textView: NSTextView, shouldChangeTextIn affectedCharRange: NSRange, replacementString: String?) -> Bool {
            guard let tv = textView as? CustomInlineNSTextView else { return true }
            
            if isProcessingAcceptOrDismiss { return true }

            if let ghostRange = tv.currentGhostTextRange {
                if NSMaxRange(affectedCharRange) > ghostRange.location || affectedCharRange.location == ghostRange.location {
                    if let replacementString = replacementString,
                       !replacementString.isEmpty &&
                       tv.ghostText()?.hasPrefix(replacementString) == true &&
                       affectedCharRange.location == ghostRange.location &&
                       affectedCharRange.length == 0 {
                        tv.consumeGhostText(length: replacementString.utf16.count)
                        
                        let newCommittedStr = tv.committedText()
                        if parent.text != newCommittedStr {
                            isInternallyUpdatingTextBinding = true
                            parent.text = newCommittedStr
                            DispatchQueue.main.async { self.isInternallyUpdatingTextBinding = false }
                        }
                        self.currentCommittedText = newCommittedStr
                        
                        parent.debouncer.cancel()
                        return false
                    } else {
                        tv.clearGhostText()
                        llmEngine.abortCurrentSuggestion()
                        parent.debouncer.cancel()
                        return true
                    }
                }
                else if NSMaxRange(affectedCharRange) <= ghostRange.location {
                    tv.clearGhostText()
                    llmEngine.abortCurrentSuggestion()
                    parent.debouncer.cancel()
                    return true
                }
            }
            return true
        }
        
        func textViewDidChangeSelection(_ notification: Notification) {
            guard let textView = notification.object as? CustomInlineNSTextView else { return }
            
            let selectedRange = textView.selectedRange
            let charIndex: Int
            
            if selectedRange.length > 0 {
                charIndex = selectedRange.location
            } else {
                charIndex = textView.lastMouseUpCharIndex ?? selectedRange.location
            }
            
            caretCoordinator?.updateCaretPosition(for: textView, at: charIndex)
            textView.lastMouseUpCharIndex = nil
        }
        
        
        private func requestTextCompletion(for textView: CustomInlineNSTextView) {
            let currentPromptForLLM = textView.committedText()
            guard !currentPromptForLLM.isEmpty else {
                textView.clearGhostText()
                llmEngine.abortCurrentSuggestion()
                return
            }
            
            if textView.ghostText() != nil {
                textView.clearGhostText()
            }

            llmEngine.generateSuggestion(prompt: currentPromptForLLM) { [weak textView] token in
                DispatchQueue.main.async {
                    textView?.appendGhostTextToken(token)
                }
            } onComplete: { [weak textView] result in
                DispatchQueue.main.async {
                    guard let textView = textView else { return }
                    switch result {
                    case .success(let fullSuggestion):
                        if fullSuggestion.isEmpty && textView.ghostText() != nil {
                            textView.clearGhostText()
                        }
                    case .failure(let error):
                        if case LLMError.aborted = error {
                        } else {
                            textView.clearGhostText()
                        }
                    }
                }
            }
        }
        
        private func handleTextChange(for textView: CustomInlineNSTextView) {
            let previousCommittedTextInCoordinator = self.currentCommittedText
            let newCommittedTextFromTextView = textView.committedText()
            
            updateTextBinding(with: newCommittedTextFromTextView)
            
            if newCommittedTextFromTextView.isEmpty {
                clearAllCompletions(for: textView)
                self.currentCommittedText = newCommittedTextFromTextView
                return
            }
            
            let committedTextChangedByUser = newCommittedTextFromTextView != previousCommittedTextInCoordinator
            let ghostTextVanished = textView.ghostText() == nil

            let shouldFetch = committedTextChangedByUser || (ghostTextVanished && !isProcessingAcceptOrDismiss)
            
            if shouldFetch {
                parent.debouncer.debounce { [weak self, weak textView] in
                    guard let self = self, let textView = textView else { return }
                    self.requestTextCompletion(for: textView)
                }
            }
            
            self.currentCommittedText = newCommittedTextFromTextView
        }
        
        private func updateTextBinding(with newText: String) {
            if parent.text != newText {
                isInternallyUpdatingTextBinding = true
                parent.text = newText
                DispatchQueue.main.async { self.isInternallyUpdatingTextBinding = false }
            }
        }
        
        private func clearAllCompletions(for textView: CustomInlineNSTextView) {
            textView.clearGhostText()
        }
        
        func updateCaret() {
            if let textView = managedTextView {
                caretCoordinator?.updateCaretPosition(for: textView)
            }
        }
    }
}

extension InlineSuggestingTextView.Coordinator: TextFormattingDelegate {
    /// Helper to toggle a symbolic font trait (bold, italic)
    private func toggleSymbolicTrait(_ trait: NSFontDescriptor.SymbolicTraits) {
        guard let textView = managedTextView else { return }
        let selectedRange = textView.selectedRange
        let applyTrait: (NSFont) -> NSFont = { font in
            var traits = font.fontDescriptor.symbolicTraits
            if traits.contains(trait) { traits.remove(trait) } else { traits.insert(trait) }
            return NSFont(descriptor: font.fontDescriptor.withSymbolicTraits(traits), size: font.pointSize) ?? font
        }
        if selectedRange.length == 0 {
            if let font = textView.typingAttributes[.font] as? NSFont {
                textView.typingAttributes[.font] = applyTrait(font)
            }
        } else {
            textView.textStorage?.enumerateAttribute(.font, in: selectedRange) { value, range, _ in
                if let font = value as? NSFont {
                    textView.textStorage?.addAttribute(.font, value: applyTrait(font), range: range)
                }
            }
        }
    }

     func toggleBold() {
         toggleSymbolicTrait(.bold)
     }

     func toggleItalic() {
         toggleSymbolicTrait(.italic)
     }

    /// Generic helper to toggle an attribute with different toggled/default values
    private func toggleAttribute<T: Equatable>(_ key: NSAttributedString.Key, toggledValue: T, defaultValue: T) {
        guard let textView = managedTextView else { return }
        let selectedRange = textView.selectedRange
        
        let apply: (T?) -> T = { current in
            (current == toggledValue) ? defaultValue : toggledValue
        }
        
        if selectedRange.length == 0 {
            let current = textView.typingAttributes[key] as? T
            textView.typingAttributes[key] = apply(current)
        } else {
            textView.textStorage?.enumerateAttribute(key, in: selectedRange) { value, range, _ in
                let current = value as? T
                let newValue = apply(current)
                textView.textStorage?.addAttribute(key, value: newValue, range: range)
            }
        }
    }

    func toggleUnderline() {
        toggleAttribute(.underlineStyle,
                        toggledValue: NSUnderlineStyle.single.rawValue,
                        defaultValue: 0)
    }

    func toggleStrikethrough() {
        toggleAttribute(.strikethroughStyle,
                        toggledValue: NSUnderlineStyle.single.rawValue,
                        defaultValue: 0)
    }

    func toggleHighlight() {
        toggleAttribute(.backgroundColor,
                        toggledValue: NSColor.yellow,
                        defaultValue: NSColor.clear)
    }
    
    /// Generic helper for line-based operations
    private func modifyLine(_ modifier: (String) -> String) {
        guard let textView = managedTextView else { return }
        let selectedRange = textView.selectedRange
        let text = textView.string as NSString
        
        let lineRange = text.lineRange(for: selectedRange)
        let line = text.substring(with: lineRange)
        let newLine = modifier(line)
        
        textView.textStorage?.replaceCharacters(in: lineRange, with: newLine)
    }

    func toggleBulletList() {
        modifyLine { line in
            line.hasPrefix("• ") ? String(line.dropFirst(2)) : "• " + line
        }
    }
    
    func toggleNumberedList() {
        modifyLine { line in
            let numberPattern = #"^\d+\.\s+"#
            if line.range(of: numberPattern, options: .regularExpression) != nil {
                return line.replacingOccurrences(of: numberPattern, with: "", options: .regularExpression)
            } else {
                return "1. " + line
            }
        }
    }
    
    func setTextAlignment(_ alignment: NSTextAlignment) {
        modifyAttribute(.paragraphStyle) { (current: NSParagraphStyle?) -> NSParagraphStyle in
            let paragraphStyle = (current?.mutableCopy() as? NSMutableParagraphStyle) ?? NSMutableParagraphStyle()
            paragraphStyle.alignment = alignment
            return paragraphStyle
        }
    }
    
    
    /// Generic helper for font operations
    private func modifyFont(_ modifier: (NSFont) -> NSFont) {
        guard let textView = managedTextView else { return }
        let selectedRange = textView.selectedRange
        
        if selectedRange.length == 0 {
            if let font = textView.typingAttributes[.font] as? NSFont {
                textView.typingAttributes[.font] = modifier(font)
            }
        } else {
            textView.textStorage?.enumerateAttribute(.font, in: selectedRange) { value, range, _ in
                if let font = value as? NSFont {
                    textView.textStorage?.addAttribute(.font, value: modifier(font), range: range)
                }
            }
        }
    }

    func increaseFontSize() {
        modifyFont { font in
            let newSize = min(font.pointSize + FontManager.shared.fontSizeStep, FontManager.shared.maxFontSize)
            return NSFont(descriptor: font.fontDescriptor, size: newSize) ?? font
        }
    }
    
    func decreaseFontSize() {
        modifyFont { font in
            let newSize = max(font.pointSize - FontManager.shared.fontSizeStep, FontManager.shared.minFontSize)
            return NSFont(descriptor: font.fontDescriptor, size: newSize) ?? font
        }
    }
    
    /// Generic helper for attribute operations
    private func modifyAttribute<T>(_ key: NSAttributedString.Key, _ modifier: (T?) -> T) {
        guard let textView = managedTextView else { return }
        let selectedRange = textView.selectedRange
        
        if selectedRange.length == 0 {
            let current = textView.typingAttributes[key] as? T
            textView.typingAttributes[key] = modifier(current)
        } else {
            textView.textStorage?.enumerateAttribute(key, in: selectedRange) { value, range, _ in
                let current = value as? T
                textView.textStorage?.addAttribute(key, value: modifier(current), range: range)
            }
        }
    }

    func setFont(_ font: NSFont) {
        modifyAttribute(.font) { _ in font }
    }
}

extension InlineSuggestingTextView.Coordinator: LLMInteractionDelegate {
    func acceptSuggestion() {
        guard let textView = managedTextView, textView.ghostText() != nil else { return }
        performSuggestionAcceptance(for: textView)
    }

    func dismissSuggestion() {
        guard let textView = managedTextView, textView.ghostText() != nil else { return }
        performSuggestionDismissal(for: textView)
    }
    
    private func performSuggestionAcceptance(for textView: CustomInlineNSTextView) {
        isProcessingAcceptOrDismiss = true
        let acceptedRange = textView.currentGhostTextRange
        textView.acceptGhostText()
        clearAllCompletions(for: textView)

        let newCommittedStr = textView.string
        updateTextBinding(with: newCommittedStr)
        currentCommittedText = newCommittedStr
        if let range = acceptedRange {
            let caretPos = range.location + range.length
            textView.selectedRange = NSRange(location: caretPos, length: 0)
            textView.scrollRangeToVisible(textView.selectedRange)
        } else {
            let end = newCommittedStr.utf16.count
            textView.selectedRange = NSRange(location: end, length: 0)
            textView.scrollRangeToVisible(textView.selectedRange)
        }

        isProcessingAcceptOrDismiss = false
        DispatchQueue.main.async {
            textView.window?.makeFirstResponder(textView)
        }
    }
    
    private func performSuggestionDismissal(for textView: CustomInlineNSTextView) {
        isProcessingAcceptOrDismiss = true
        
        clearAllCompletions(for: textView)
        
        isProcessingAcceptOrDismiss = false
        
        DispatchQueue.main.async {
            textView.window?.makeFirstResponder(textView)
        }
    }
}

class CustomInlineNSTextView: NSTextView {
    weak var llmInteractionDelegate: LLMInteractionDelegate?
    private var customLayoutManager: CustomLayoutManager!
    var currentGhostTextRange: NSRange?
    private var lastCommittedTextForChangeDetection: String = ""
    var lastMouseUpCharIndex: Int? = nil

    override func viewDidMoveToWindow() {
        super.viewDidMoveToWindow()
        notifyDelegate()
    }

    override var shouldDrawInsertionPoint: Bool {
        return false
    }

    override init(frame frameRect: NSRect, textContainer container: NSTextContainer?) {
        let textStorage = NSTextStorage()
        customLayoutManager = CustomLayoutManager()
        textStorage.addLayoutManager(customLayoutManager)
        let newTextContainer = container ?? NSTextContainer(size: CGSize(width: frameRect.width, height: CGFloat.greatestFiniteMagnitude))
        if container == nil {
            newTextContainer.widthTracksTextView = true
        }
        customLayoutManager.addTextContainer(newTextContainer)
        super.init(frame: frameRect, textContainer: newTextContainer)
        if container == nil {
            self.minSize = NSSize(width: 0, height: frameRect.height)
            self.maxSize = NSSize(width: CGFloat.greatestFiniteMagnitude, height: CGFloat.greatestFiniteMagnitude)
            self.isVerticallyResizable = true
            self.isHorizontallyResizable = false
            self.autoresizingMask = [.width]
        }
        self.lastCommittedTextForChangeDetection = self.string
    }
    
    convenience override init(frame frameRect: NSRect) {
         self.init(frame: frameRect, textContainer: nil)
    }

    required init?(coder: NSCoder) {
        fatalError("init(coder:) has not been implemented")
    }

    override func updateTrackingAreas() {
        super.updateTrackingAreas()
    }
    override func mouseMoved(with event: NSEvent) {
        super.mouseMoved(with: event)
    }
    override func mouseEntered(with event: NSEvent) {
        super.mouseEntered(with: event)
    }
    override func mouseExited(with event: NSEvent) {
        super.mouseExited(with: event)
    }

    override func mouseDown(with event: NSEvent) {
        if hasGhostText() {
            llmInteractionDelegate?.dismissSuggestion()
        }
        
        if let coordinator = delegate as? InlineSuggestingTextView.Coordinator {
            coordinator.caretCoordinator?.setExpanded(false)
        }
        
        super.mouseDown(with: event)
        if let coordinator = delegate as? InlineSuggestingTextView.Coordinator {
            DispatchQueue.main.async {
                coordinator.parent.formattingDelegate = coordinator
                coordinator.caretCoordinator?.updateCaretPosition(for: self)
            }
        }
    }

    override func mouseUp(with event: NSEvent) {
        super.mouseUp(with: event)
        notifyDelegate()
    }
    
    override func keyDown(with event: NSEvent) {
        // Intercept Tab to accept suggestion, Escape to dismiss
        if event.keyCode == KeyCodes.tab {
            llmInteractionDelegate?.acceptSuggestion()
            return
        }
        if event.keyCode == KeyCodes.escape {
            llmInteractionDelegate?.dismissSuggestion()
            return
        }
        super.keyDown(with: event)
        notifyDelegate()
    }
    
    private func notifyDelegate() {
        if let coord = delegate as? InlineSuggestingTextView.Coordinator {
            coord.updateCaret()
        }
    }

    override var string: String {
        didSet {
            if !self.hasGhostText() {
                self.lastCommittedTextForChangeDetection = super.string
            }
        }
    }

    // Ensure delegate textDidChange is always called and caret updates after typing
    override func didChangeText() {
        super.didChangeText()
        notifyDelegate()
    }

    func didCommittedTextChangeByUser(newCommittedText: String, previousCommittedText: String) -> Bool {
        return newCommittedText != previousCommittedText
    }

    func committedText() -> String {
        guard let ts = self.textStorage else { return "" }
        if let ghostRange = currentGhostTextRange, ghostRange.location <= ts.length {
            return (ts.string as NSString).substring(to: ghostRange.location)
        }
        return ts.string
    }
    
    func ghostText() -> String? {
        guard let ts = self.textStorage else { return nil }
        if let ghostRange = currentGhostTextRange, NSMaxRange(ghostRange) <= ts.length {
            if ghostRange.location >= 0 && ghostRange.length >= 0 && NSMaxRange(ghostRange) <= ts.string.utf16.count {
                return (ts.string as NSString).substring(with: ghostRange)
            } else {
                currentGhostTextRange = nil
                return nil
            }
        }
        return nil
    }
    
    func hasGhostText() -> Bool {
        if let range = currentGhostTextRange, range.length > 0, let storage = textStorage {
            return NSMaxRange(range) <= storage.length && range.location <= storage.length
        }
        return false
    }

    func appendGhostTextToken(_ token: String) {
        guard let ts = self.textStorage, !token.isEmpty else { return }

        ts.beginEditing()
        let attributes: [NSAttributedString.Key: Any] = [
            .isGhostText: true,
            .foregroundColor: NSColor.gray,
            .font: self.font ?? NSFont.systemFont(ofSize: 16)
        ]
        
        let insertionPointForNewSuggestion = self.selectedRange.location

        if let existingRange = currentGhostTextRange,
           existingRange.location <= ts.length && NSMaxRange(existingRange) <= ts.length {
            
            if insertionPointForNewSuggestion != existingRange.location {
                if NSMaxRange(existingRange) <= ts.length {
                    ts.replaceCharacters(in: existingRange, with: "")
                }
                currentGhostTextRange = nil
            } else {
                let appendLocation = NSMaxRange(existingRange)
                guard appendLocation <= ts.length else {
                    ts.endEditing()
                    return
                }
                ts.insert(NSAttributedString(string: token, attributes: attributes), at: appendLocation)
                currentGhostTextRange = NSRange(location: existingRange.location, length: existingRange.length + token.utf16.count)
            }
        }
        
        if currentGhostTextRange == nil {
            ts.insert(NSAttributedString(string: token, attributes: attributes), at: insertionPointForNewSuggestion)
            currentGhostTextRange = NSRange(location: insertionPointForNewSuggestion, length: token.utf16.count)
        }
        ts.endEditing()
        
        if let finalGhostRange = currentGhostTextRange {
            self.selectedRange = NSRange(location: finalGhostRange.location, length: 0)
            self.scrollRangeToVisible(finalGhostRange)
        }
        self.typingAttributes[.foregroundColor] = NSColor.textColor
    }

    func clearGhostText() {
        guard let ts = self.textStorage, let ghostRange = currentGhostTextRange, ghostRange.length > 0 else {
            currentGhostTextRange = nil
            return
        }
        if ghostRange.location <= ts.length && NSMaxRange(ghostRange) <= ts.length {
            ts.beginEditing()
            ts.replaceCharacters(in: ghostRange, with: "")
            ts.endEditing()
        }
        currentGhostTextRange = nil
    }

    func acceptGhostText() {
        guard let ts = self.textStorage, let ghostRange = currentGhostTextRange, ghostRange.length > 0 else { return }
        
        if ghostRange.location <= ts.length && NSMaxRange(ghostRange) <= ts.length {
            let normalAttributes: [NSAttributedString.Key: Any] = [
                .foregroundColor: NSColor.textColor,
                .font: self.font ?? NSFont.systemFont(ofSize: 16)
            ]
            ts.beginEditing()
            ts.removeAttribute(.isGhostText, range: ghostRange)
            ts.addAttributes(normalAttributes, range: ghostRange)
            ts.endEditing()
            
            self.lastCommittedTextForChangeDetection = ts.string
        }
        currentGhostTextRange = nil
    }
    
    func consumeGhostText(length: Int) {
        guard let ts = self.textStorage, let ghostRange = currentGhostTextRange, length > 0 else { return }

        if length <= ghostRange.length {
            let consumedRange = NSRange(location: ghostRange.location, length: length)
            let remainingGhostLength = ghostRange.length - length
            
            ts.beginEditing()
            let normalAttributes: [NSAttributedString.Key: Any] = [
                .foregroundColor: NSColor.textColor,
                .font: self.font ?? NSFont.systemFont(ofSize: 16)
            ]
            ts.removeAttribute(.isGhostText, range: consumedRange)
            ts.addAttributes(normalAttributes, range: consumedRange)
            
            if remainingGhostLength > 0 {
                currentGhostTextRange = NSRange(location: ghostRange.location + length, length: remainingGhostLength)
            } else {
                currentGhostTextRange = nil
            }
        ts.endEditing()

        let newLocation = NSMaxRange(consumedRange)
        self.selectedRange = NSRange(location: newLocation, length: 0)
        self.scrollRangeToVisible(self.selectedRange)
        
        self.lastCommittedTextForChangeDetection = self.committedText()
        } else {
            acceptGhostText()
        }
    }
}

class CustomLayoutManager: NSLayoutManager {
    override func drawGlyphs(forGlyphRange glyphsToShow: NSRange, at origin: NSPoint) {
        super.drawGlyphs(forGlyphRange: glyphsToShow, at: origin)
    }
}

class Debouncer: ObservableObject {
    private let delay: TimeInterval
    private var workItem: DispatchWorkItem?
    private let queue: DispatchQueue
    init(delay: TimeInterval, queue: DispatchQueue = DispatchQueue.main) {
        self.delay = delay
        self.queue = queue
    }
    func debounce(action: @escaping () -> Void) {
        workItem?.cancel()
        let newWorkItem = DispatchWorkItem(block: action)
        workItem = newWorkItem
        queue.asyncAfter(deadline: .now() + delay, execute: newWorkItem)
    }
    func cancel() {
        workItem?.cancel()
    }
}

extension NSFont {
    func withSymbolicTraits(_ traits: NSFontDescriptor.SymbolicTraits) -> NSFont? {
        let descriptorOptional: NSFontDescriptor? = fontDescriptor.withSymbolicTraits(traits)
        guard let descriptor = descriptorOptional else {
            return nil
        }
        return NSFont(descriptor: descriptor, size: pointSize)
    }
    var isBold: Bool {
        return fontDescriptor.symbolicTraits.contains(.bold)
    }
    var isItalic: Bool {
        return fontDescriptor.symbolicTraits.contains(.italic)
    }
}

enum LLMError: Error {
    case aborted
    case processLaunchError(String)
    case engineNotRunning
    case pythonScriptNotReady
    case promptEncodingError
    case stdinWriteError(String)
    case scriptError(String)
    case other(Error)
}

#if canImport(Carbon)
import Carbon.HIToolbox
#endif

enum KeyCodes {
    static let tab: UInt16 = 0x30
    static let rightArrow: UInt16 = 0x7C
    static let escape: UInt16 = 0x35
}


import Combine
import SwiftUI
import AppKit
import QuartzCore

struct InlineSuggestingTextView: NSViewRepresentable {
    @Binding var text: String
    @ObservedObject var llmEngine: LLMEngine
    @ObservedObject var audioEngine: AudioEngine
    @Binding var debouncer: Debouncer
    @Binding var formattingDelegate: TextFormattingDelegate?
    var onSelectionChange: (() -> Void)? = nil
    var onCoordinatorChange: ((CaretUICoordinator) -> Void)? = nil
    @Binding var viewUpdater: Bool
    
    private let fontManager = FontManager.shared

    func makeCoordinator() -> Coordinator {
        Coordinator(self, llmEngine: llmEngine, audioEngine: audioEngine)
    }

    func makeNSView(context: Context) -> NSScrollView {
        let textView = CustomInlineNSTextView(frame: .zero)
        
        textView.delegate = context.coordinator
        textView.font = fontManager.defaultEditorFont()
        
        let defaultPS = NSMutableParagraphStyle()
        defaultPS.alignment = .left
        textView.defaultParagraphStyle = defaultPS
        textView.typingAttributes[.paragraphStyle] = defaultPS
        textView.isEditable = true
        textView.isSelectable = true
        textView.allowsUndo = true
        textView.textContainerInset = CGSize(width: 10, height: 80)
        
        textView.drawsBackground = false 
        textView.backgroundColor = .clear
        // Hide the default system caret
        textView.insertionPointColor = .clear
    
        textView.llmInteractionDelegate = context.coordinator
        context.coordinator.managedTextView = textView

        setupTextViewContent(textView, context: context)
        
        let scrollView = createScrollView(with: textView)
        
        setupCustomCaret(textView, context: context)
        
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
        let caretCoordinator = CaretUICoordinator(llmEngine: llmEngine, audioEngine: audioEngine)
        context.coordinator.caretCoordinator = caretCoordinator
        onCoordinatorChange?(caretCoordinator)

        DispatchQueue.main.async {
            caretCoordinator.updateCaretPosition(for: textView)
        }
    }

    func updateNSView(_ nsView: NSScrollView, context: Context) {
        DispatchQueue.main.async {
            formattingDelegate = context.coordinator
            FormattingCommands.shared.delegate = context.coordinator
        }
        
        guard let textView = nsView.documentView as? CustomInlineNSTextView else { return }
        // Ensure system caret remains hidden
        textView.insertionPointColor = .clear
    
        if textView.committedText() != text && !context.coordinator.isInternallyUpdatingTextBinding {
            textView.clearGhostText()
            llmEngine.abortSuggestion(for: "autocomplete")
            textView.string = text
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

        init(_ parent: InlineSuggestingTextView, llmEngine: LLMEngine, audioEngine: AudioEngine) {
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
    
        private enum FormattingCheck {
            case symbolicTrait(NSFontDescriptor.SymbolicTraits)
            case attribute(key: NSAttributedString.Key, value: AnyHashable)
            case paragraphList(marker: NSTextList.MarkerFormat)
        }

    
        private func isActive(_ check: FormattingCheck) -> Bool {
            guard let textView = managedTextView else { return false }
            let range = textView.selectedRange

            switch check {
            case .symbolicTrait(let trait):
                return fontTraitActive(in: range, trait: trait)
            case .attribute(let key, let value):
                return attributeActive( in: range, key: key, value: value )
            case .paragraphList(let marker):
                return paragraphListActive(in: range, markerFormat: marker)
            }
        }

    
        func isBoldActive() -> Bool {
            return isActive(.symbolicTrait(.bold))
        }
    
        func isItalicActive() -> Bool {
            return isActive(.symbolicTrait(.italic))
        }
    
        func isUnderlineActive() -> Bool {
            return isActive(.attribute(key: .underlineStyle,
                                       value: NSUnderlineStyle.single.rawValue))
        }
    
        func isStrikethroughActive() -> Bool {
            return isActive(.attribute(key: .strikethroughStyle,
                                       value: NSUnderlineStyle.single.rawValue))
        }
            
        private func paragraphListActive(in range: NSRange,
                                         markerFormat: NSTextList.MarkerFormat) -> Bool {
            guard let textView = managedTextView else { return false }
            let nsText = textView.string as NSString

            let pattern: String
            switch markerFormat {
            case .disc:
                // Check for the specific bullet we are using.
                pattern = #"^\s*•\s+"#
            case .decimal:
                // Check for any number followed by a dot.
                pattern = #"^\s*\d+\.\s+"#
            default:
                return false
            }

            // Function to check a single paragraph
            let checkParagraph = { (paraRange: NSRange) -> Bool in
                let paragraph = nsText.substring(with: paraRange)
                return paragraph.range(of: pattern, options: .regularExpression) != nil
            }

            // 1) If there is a selection, check if any paragraph in the selection has the list format
            if range.length > 0 {
                var found = false
                nsText.enumerateSubstrings(in: range, options: .byParagraphs) { _, paraRange, _, stop in
                    if !nsText.substring(with: paraRange).trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
                        if checkParagraph(paraRange) {
                            found = true
                            stop.pointee = true
                        }
                    }
                }
                return found
            }

            // 2) No selection, check the current paragraph under the cursor
            let loc = max(range.location, 0)
            let paraRange = nsText.paragraphRange(for: NSRange(location: loc, length: 0))
            return checkParagraph(paraRange)
        }


        func isBulletListActive() -> Bool {
            // if managedTextView is nil, fall back to empty range (0,0)
            let range = managedTextView?.selectedRange ?? NSRange(location: 0, length: 0)
            return paragraphListActive(in: range, markerFormat: .disc)
        }

        func isNumberedListActive() -> Bool {
            let range = managedTextView?.selectedRange ?? NSRange(location: 0, length: 0)
            return paragraphListActive(in: range, markerFormat: .decimal)
        }

        // MARK: – Проверка шрифтовых traits (bold/italic)Add commentMore actions
        private func fontTraitActive(in range: NSRange,
                                     trait: NSFontDescriptor.SymbolicTraits) -> Bool {
            guard let textView = managedTextView else { return false }

            // Если есть выделение — считаем active, если хотя бы один символ в нём имеет trait
            if range.length > 0 {
                var found = false
                textView.textStorage?.enumerateAttribute(.font, in: range) { value, _, stop in
                    if let font = value as? NSFont,
                       font.fontDescriptor.symbolicTraits.contains(trait) {
                        found = true
                        stop.pointee = true
                    }
                }
                return found
            }

            // Нет выделения — смотрим typingAttributes, как раньше
            if let typingFont = textView.typingAttributes[.font] as? NSFont {
                return typingFont.fontDescriptor.symbolicTraits.contains(trait)
            }
            return false
        }

        // MARK: – Проверка текстовых атрибутов (underline/strikethrough)
        private func attributeActive(in range: NSRange,
                                     key: NSAttributedString.Key,
                                     value: AnyHashable) -> Bool {
            guard let textView = managedTextView else { return false }

            // Есть выделение — true, если хотя бы один символ в нём имеет атрибут = value
            if range.length > 0 {
                var found = false
                textView.textStorage?.enumerateAttribute(key, in: range) { current, _, stop in
                    if let h = current as? AnyHashable, h == value {
                        found = true
                        stop.pointee = true
                    }
                }
                return found
            }

            // Нет выделения — только typingAttributes
            if let h = textView.typingAttributes[key] as? AnyHashable {
                return h == value
            }
            return false
        }
        
        private func isAlignmentActive(_ alignment: NSTextAlignment) -> Bool {
            guard let tv = managedTextView else { return false }
            let range = tv.selectedRange
            
            // 1) Если есть выделение — смотрим стиль первого символа
            if range.length > 0,
               let ps = tv.textStorage?
                     .attribute(.paragraphStyle, at: range.location, effectiveRange: nil)
                     as? NSParagraphStyle {
                return ps.alignment == alignment
            }
            // 2) Нет выделения — смотрим typingAttributes
            if let ps = tv.typingAttributes[.paragraphStyle] as? NSParagraphStyle {
                return ps.alignment == alignment
            }
            return alignment == .left
        }
        
        func isLeftAlignActive() -> Bool { isAlignmentActive(.left) }
        func isCenterAlignActive() -> Bool { isAlignmentActive(.center) }
        func isRightAlignActive() -> Bool { isAlignmentActive(.right) }

        func textDidChange(_ notification: Notification) {
            guard let textView = notification.object as? CustomInlineNSTextView else { return }
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
                        llmEngine.abortSuggestion(for: "autocomplete")
                        parent.debouncer.cancel()
                        return true
                    }
                }
                else if NSMaxRange(affectedCharRange) <= ghostRange.location {
                    tv.clearGhostText()
                    llmEngine.abortSuggestion(for: "autocomplete")
                    parent.debouncer.cancel()
                    return true
                }
            }
            return true
        }
        
        func textViewDidChangeSelection(_ notification: Notification) {
            guard let textView = notification.object as? CustomInlineNSTextView else { return }
            
            if textView.ghostText() != nil {
                return
            }
            
            let selectedRange = textView.selectedRange
            let charIndex: Int = textView.lastMouseUpCharIndex ?? selectedRange.location
            caretCoordinator?.updateCaretPosition(for: textView, at: charIndex)
            textView.lastMouseUpCharIndex = nil
            DispatchQueue.main.async { [weak self] in
                guard let self = self else { return }
                self.parent.onSelectionChange?()
                self.parent.viewUpdater.toggle()
            }
        }
        
        func requestTextCompletion(for textView: CustomInlineNSTextView) {
            let currentPromptForLLM = textView.committedText()
            guard !currentPromptForLLM.isEmpty else {
                textView.clearGhostText()
                llmEngine.abortSuggestion(for: "autocomplete")
                return
            }
            
            if textView.ghostText() != nil {
                textView.clearGhostText()
            }

            llmEngine.generateSuggestion(
                for: "autocomplete",
                prompt: currentPromptForLLM) { [weak self, weak textView] token in
                DispatchQueue.main.async {
                    textView?.appendGhostTextToken(token)
                    // Trigger caret effect for each suggestion token
                    self?.caretCoordinator?.triggerCaretEffect = true
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
                        if case LLMEngine.LLMError.aborted = error {
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

final class FormattingCommands {
    static let shared = FormattingCommands()
    weak var delegate: TextFormattingDelegate?   // <-- это твой делегат, обычно Coordinator
    private init() {}

    func toggleBold() { delegate?.toggleBold() }
    func toggleItalic() { delegate?.toggleItalic() }
    func toggleUnderline() { delegate?.toggleUnderline() }
    func toggleStrikethrough() { delegate?.toggleStrikethrough() }
}


extension InlineSuggestingTextView.Coordinator: TextFormattingDelegate {
    func toggleSymbolicTrait(_ trait: NSFontDescriptor.SymbolicTraits) {
        guard let textView = managedTextView else { return }
        let range = textView.selectedRange

        // 1) Определяем, хотим ли мы убрать или добавить
        let shouldRemove = fontTraitActive(in: range, trait: trait)

        // 2) Функция, которая либо убирает, либо добавляет
        let applyTrait: (NSFont) -> NSFont = { font in
            var traits = font.fontDescriptor.symbolicTraits
            if shouldRemove {
                traits.remove(trait)
            } else {
                traits.insert(trait)
            }
            return NSFont(descriptor: font.fontDescriptor.withSymbolicTraits(traits), size: font.pointSize) ?? font
        }

        // 3) Применяем ко всему выделению или к typingAttributes
        if range.length > 0 {
            textView.textStorage?.enumerateAttribute(.font, in: range) { value, subRange, _ in
                if let f = value as? NSFont {
                    textView.textStorage?.addAttribute(.font, value: applyTrait(f), range: subRange)
                }
            }
        } else {
            // пустой курсор — изменяем typingAttributes
            if let f = textView.typingAttributes[.font] as? NSFont {
                textView.typingAttributes[.font] = applyTrait(f)
            }
        }

        // 4) Обновляем тулбар сразу после клика
        DispatchQueue.main.async { [weak self] in
            self?.parent.onSelectionChange?()
        }
    }

    func toggleAttribute(_ key: NSAttributedString.Key,
                         toggledValue: AnyHashable) {
        guard let textView = managedTextView else { return }
        let range = textView.selectedRange

        // 1) Определяем, убрать ли из выделения
        let shouldRemove = attributeActive(in: range, key: key, value: toggledValue)

        // 2) Применяем ко всему выделению или к typingAttributes
        if range.length > 0 {
            textView.textStorage?.enumerateAttribute(key, in: range) { current, subRange, _ in
                if shouldRemove {
                    textView.textStorage?.removeAttribute(key, range: subRange)
                } else {
                    textView.textStorage?.addAttribute(key, value: toggledValue, range: subRange)
                }
            }
        } else {
            // пустой курсор — изменяем typingAttributes
            if shouldRemove {
                textView.typingAttributes.removeValue(forKey: key)
            } else {
                textView.typingAttributes[key] = toggledValue
            }
        }

        DispatchQueue.main.async { [weak self] in
            self?.parent.onSelectionChange?()
        }
    }

    
    func toggleBold() {
        toggleSymbolicTrait(.bold)

    }

    func toggleItalic() {
        toggleSymbolicTrait(.italic)
    }

    func toggleUnderline() {
        toggleAttribute(.underlineStyle,
                        toggledValue: NSUnderlineStyle.single.rawValue)
    }

    func toggleStrikethrough() {
        toggleAttribute(.strikethroughStyle,
                        toggledValue: NSUnderlineStyle.single.rawValue)
    }

    func toggleHighlight() {
        toggleAttribute(.backgroundColor,
                        toggledValue: NSColor.yellow)
    }
    
    func toggleParagraphList(marker: NSTextList.MarkerFormat) {
        guard let textView = managedTextView, let textStorage = textView.textStorage else { return }

        let selectedRange = textView.selectedRange
        let text = textView.string as NSString

        // Handle case where text view is completely empty
        if text.length == 0 {
            let prefix = (marker == .disc) ? "• " : "1. "
            let attributes = textView.typingAttributes
            let attributedPrefix = NSAttributedString(string: prefix, attributes: attributes)
            
            textStorage.beginEditing()
            textStorage.insert(attributedPrefix, at: 0)
            textStorage.endEditing()
            
            updateTextBinding(with: textView.string)
            DispatchQueue.main.async { self.parent.onSelectionChange?() }
            return
        }
        
        let rangeToFormat = text.paragraphRange(for: selectedRange)
        
        let isBulletActive = isBulletListActive()
        let isNumberedActive = isNumberedListActive()
        let shouldRemove = (marker == .disc && isBulletActive) || (marker == .decimal && isNumberedActive)

        var paragraphRanges: [NSRange] = []
        text.enumerateSubstrings(in: rangeToFormat, options: .byParagraphs) { _, subRange, _, _ in
            // Include the paragraph even if it's empty, as long as it's part of the selection
            paragraphRanges.append(subRange)
        }

        // If cursor is at an empty paragraph at the very end (enumeration returns nothing)
        if paragraphRanges.isEmpty {
            paragraphRanges.append(NSRange(location: selectedRange.location, length: 0))
        }

        textStorage.beginEditing()
        
        if shouldRemove {
            let pattern = (marker == .disc) ? #"^\s*•\s+"# : #"^\s*\d+\.\s+"#
            for range in paragraphRanges.reversed() {
                let paragraphText = text.substring(with: range)
                if let match = paragraphText.range(of: pattern, options: .regularExpression) {
                    let absoluteRange = NSRange(location: range.location + match.lowerBound.utf16Offset(in: paragraphText),
                                                length: match.upperBound.utf16Offset(in: paragraphText) - match.lowerBound.utf16Offset(in: paragraphText))
                    textStorage.replaceCharacters(in: absoluteRange, with: "")
                }
            }
        } else { // Add/change formatting
            // In reverse order to not invalidate ranges
            for range in paragraphRanges.reversed() {
                // First, strip any other list type to avoid mixing formats
                let bulletPattern = #"^\s*•\s+"#
                let numberPattern = #"^\s*\d+\.\s+"#
                let paragraphText = (textStorage.string as NSString).substring(with: range)
                
                if let match = paragraphText.range(of: numberPattern, options: .regularExpression) {
                     let absoluteRange = NSRange(location: range.location + match.lowerBound.utf16Offset(in: paragraphText),
                                                 length: match.upperBound.utf16Offset(in: paragraphText) - match.lowerBound.utf16Offset(in: paragraphText))
                     textStorage.replaceCharacters(in: absoluteRange, with: "")
                }
                if let match = paragraphText.range(of: bulletPattern, options: .regularExpression) {
                    let absoluteRange = NSRange(location: range.location + match.lowerBound.utf16Offset(in: paragraphText),
                                                length: match.upperBound.utf16Offset(in: paragraphText) - match.lowerBound.utf16Offset(in: paragraphText))
                    textStorage.replaceCharacters(in: absoluteRange, with: "")
                }

                // Add the new prefix
                let prefix = (marker == .disc) ? "• " : "1. "
                let attributes: [NSAttributedString.Key: Any]
                if range.location < textStorage.length {
                    attributes = textStorage.attributes(at: range.location, effectiveRange: nil)
                } else {
                    attributes = textView.typingAttributes
                }
                let attributedPrefix = NSAttributedString(string: prefix, attributes: attributes)
                textStorage.insert(attributedPrefix, at: range.location)
            }
        }
        
        textStorage.endEditing()
        
        // After all edits, re-number if it's a numbered list
        if !shouldRemove && marker == .decimal {
            renumberList(in: rangeToFormat)
        }

        updateTextBinding(with: textView.string)
        
        DispatchQueue.main.async { [weak self] in
            self?.parent.onSelectionChange?()
            self?.caretCoordinator?.updateCaretPosition(for: textView)
        }
    }
    
    private func renumberList(in range: NSRange) {
        guard let textView = managedTextView, let textStorage = textView.textStorage else { return }
        
        let text = textStorage.string as NSString
        var paragraphRanges: [NSRange] = []
        
        let listParagraphRange = text.paragraphRange(for: range)
        
        text.enumerateSubstrings(in: listParagraphRange, options: .byParagraphs) { _, subRange, _, _ in
            // Include empty paragraphs in re-numbering scan
            paragraphRanges.append(subRange)
        }
        
        textStorage.beginEditing()
        
        var currentNumber = 1
        let numberPattern = #"^\s*(\d+)\.\s+"#
        
        for pRange in paragraphRanges {
            let paragraphText = text.substring(with: pRange)
            
            if let regex = try? NSRegularExpression(pattern: numberPattern) {
                if let match = regex.firstMatch(in: paragraphText, options: [], range: NSRange(location: 0, length: paragraphText.utf16.count)) {
                    let newMarker = "\(currentNumber). "
                    let rangeToReplace = NSRange(location: pRange.location + match.range.location, length: match.range.length)
                    textStorage.replaceCharacters(in: rangeToReplace, with: newMarker)
                    currentNumber += 1
                }
            }
        }
        
        textStorage.endEditing()
    }

    func toggleBulletList() {
      toggleParagraphList(marker: .disc)
    }
    func toggleNumberedList() {
      toggleParagraphList(marker: .decimal)
    }
    
    func setTextAlignment(_ alignment: NSTextAlignment) {
        guard let tv = managedTextView else { return }
        
        // 1) Находим текущий параграф
        let ns = tv.string as NSString
        let sel = tv.selectedRange
        let paraRange = ns.paragraphRange(for: NSRange(location: sel.location, length: 0))
        
        // 2) Готовим новый NSMutableParagraphStyle
        let currentPS = (tv.typingAttributes[.paragraphStyle] as? NSParagraphStyle)?
                            .mutableCopy() as? NSMutableParagraphStyle
                        ?? NSMutableParagraphStyle()
        currentPS.alignment = alignment
        
        // 3) Применяем к тексту и typingAttributes
        tv.textStorage?.addAttribute(.paragraphStyle, value: currentPS, range: paraRange)
        tv.typingAttributes[.paragraphStyle] = currentPS
        
        // 4) Обновляем состояние кнопок и каретки
        DispatchQueue.main.async {
            self.parent.onSelectionChange?()
            self.caretCoordinator?.updateCaretPosition(for: tv)
        }
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
        caretCoordinator?.triggerBounceRight = true
        performSuggestionAcceptance(for: textView)
    }


    func dismissSuggestion() {
        guard let textView = managedTextView else { return }
        caretCoordinator?.triggerBounceLeft = true
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
        // Триггерим генерацию новой подсказки сразу после Tab
        parent.debouncer.debounce { [weak self, weak textView] in
            guard let self = self, let textView = textView else { return }
            self.requestTextCompletion(for: textView)
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
        if event.keyCode == KeyCodes.enter {
            let currentRange = self.selectedRange
            let text = self.string as NSString
            let currentLineRange = text.lineRange(for: currentRange)
            let currentLine = text.substring(with: currentLineRange)
            
            let bulletPattern = #"^\s*•\s+"#
            let numberPattern = #"^\s*(\d+)\.\s+"#
            
            // Handle bullet list continuation
            if let regex = try? NSRegularExpression(pattern: bulletPattern),
               let match = regex.firstMatch(in: currentLine, options: [], range: NSRange(location: 0, length: (currentLine as NSString).length)) {
                let marker = (currentLine as NSString).substring(with: match.range)
                let contentRange = NSRange(location: match.range.location + match.range.length, length: (currentLine as NSString).length - (match.range.location + match.range.length))
                let content = (currentLine as NSString).substring(with: contentRange).trimmingCharacters(in: .whitespacesAndNewlines)

                if content.isEmpty {
                    // Empty line with bullet, so break out of list
                    self.textStorage?.replaceCharacters(in: currentLineRange, with: "")
                    setSelectedRange(NSRange(location: currentLineRange.location, length: 0))
                } else {
                    // Continue list
                    insertText("\n" + marker, replacementRange: self.selectedRange)
                }
                return
            }
            
            // Handle numbered list continuation
            let nsCurrentLine = currentLine as NSString
            if let regex = try? NSRegularExpression(pattern: numberPattern),
               let match = regex.firstMatch(in: currentLine, options: [], range: NSRange(location: 0, length: nsCurrentLine.length)) {
                
                let numberRange = match.range(at: 1)
                let currentNumberStr = nsCurrentLine.substring(with: numberRange)
                let contentRange = NSRange(location: match.range.location + match.range.length, length: nsCurrentLine.length - (match.range.location + match.range.length))
                let content = nsCurrentLine.substring(with: contentRange).trimmingCharacters(in: .whitespacesAndNewlines)

                if content.isEmpty {
                    // Empty line with number, break out
                    self.textStorage?.replaceCharacters(in: currentLineRange, with: "")
                    setSelectedRange(NSRange(location: currentLineRange.location, length: 0))
                } else {
                    if let currentNumber = Int(currentNumberStr) {
                        let nextMarker = "\(currentNumber + 1). "
                        insertText("\n" + nextMarker, replacementRange: self.selectedRange)
                    } else {
                        super.keyDown(with: event) // Fallback
                    }
                }
                return
            }
        }

        // Intercept Tab to accept suggestion, or force-generate if none
        if event.keyCode == KeyCodes.tab {
            if self.ghostText() != nil {
                if let coordinator = delegate as? InlineSuggestingTextView.Coordinator {
                    coordinator.caretCoordinator?.triggerBounceRight = true
                }
                llmInteractionDelegate?.acceptSuggestion()
            } else if let coordinator = delegate as? InlineSuggestingTextView.Coordinator {
                coordinator.caretCoordinator?.triggerBounceRight = true
                // Force-generate suggestion for current context
                coordinator.parent.debouncer.debounce { [weak coordinator, weak self] in
                    guard let coordinator = coordinator, let self = self else { return }
                    coordinator.requestTextCompletion(for: self)
                }
            }
            return
        }
        // Intercept Escape to dismiss
        if event.keyCode == KeyCodes.escape {
            if let coordinator = delegate as? InlineSuggestingTextView.Coordinator {
                coordinator.caretCoordinator?.triggerBounceLeft = true
            }
            llmInteractionDelegate?.dismissSuggestion()
            return
        }
        // Handle Cmd+Right and Cmd+Left for custom completions/animations
        if event.modifierFlags.contains(.command) {
            if event.keyCode == KeyCodes.rightArrow {
                // Cmd+Right: regenerate suggestion, trigger bounce right
                if let coordinator = delegate as? InlineSuggestingTextView.Coordinator {
                    self.clearGhostText()
                    coordinator.caretCoordinator?.triggerBounceRight = true
                    coordinator.parent.debouncer.debounce { [weak coordinator, weak self] in
                        guard let coordinator = coordinator, let self = self else { return }
                        coordinator.requestTextCompletion(for: self)
                    }
                }
                return
            } else if event.keyCode == KeyCodes.leftArrow {
                // Cmd+Left: regenerate suggestion, trigger bounce left
                if let coordinator = delegate as? InlineSuggestingTextView.Coordinator {
                    self.clearGhostText()
                    coordinator.caretCoordinator?.triggerBounceLeft = true
                    coordinator.parent.debouncer.debounce { [weak coordinator, weak self] in
                        guard let coordinator = coordinator, let self = self else { return }
                        coordinator.requestTextCompletion(for: self)
                    }
                }
                return
            }
        }
        super.keyDown(with: event)
        notifyDelegate()
    }
    
    private func notifyDelegate() {
        // Caret position updates are handled by textViewDidChangeSelection only
        // to prevent race conditions during rapid text changes
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

            // caret должен быть в конце только что введённого текста
            let newLocation = ghostRange.location + length
            self.selectedRange = NSRange(location: newLocation, length: 0)
            self.scrollRangeToVisible(self.selectedRange)
            // Явно обновляем кастомную каретку (если есть)
            if let coordinator = self.delegate as? InlineSuggestingTextView.Coordinator {
                coordinator.caretCoordinator?.updateCaretPosition(for: self, at: newLocation)
            }
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

#if canImport(Carbon)
import Carbon.HIToolbox
#endif

enum KeyCodes {
    static let tab: UInt16 = 0x30
    static let rightArrow: UInt16 = 0x7C
    static let leftArrow: UInt16 = 0x7B
    static let escape: UInt16 = 0x35
    static let enter: UInt16 = 0x24
}

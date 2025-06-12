import SwiftUI
import AppKit

// --- NSViewRepresentable ---
struct InlineSuggestingTextView: NSViewRepresentable {
    @Binding var text: String
    @ObservedObject var llmEngine: LLMEngine
    @Binding var debouncer: Debouncer
    @Binding var formattingDelegate: TextFormattingDelegate?

    func makeCoordinator() -> Coordinator {
        Coordinator(self, llmEngine: llmEngine)
    }

    func makeNSView(context: Context) -> NSScrollView {
        let textView = CustomInlineNSTextView(frame: .zero) // Frame will be set by ScrollView
        
        // Configure the TextView
        textView.delegate = context.coordinator
        textView.font = .systemFont(ofSize: 16)
        textView.isEditable = true
        textView.isSelectable = true
        textView.allowsUndo = true
        textView.textContainerInset = CGSize(width: 10, height: 10)
        
        // Make the TextView background transparent to blend with the window
        textView.backgroundColor = .clear
        textView.drawsBackground = false // Ensure it doesn't draw its own background

        textView.llmInteractionDelegate = context.coordinator
        context.coordinator.managedTextView = textView // Give coordinator a reference

        // Initial text setup
        if !text.isEmpty {
             textView.string = text
        }
        context.coordinator.currentCommittedText = text

        // Setup ScrollView
        let scrollView = NSScrollView()
        scrollView.hasVerticalScroller = true
        scrollView.borderType = .noBorder
        scrollView.documentView = textView
        
        // Make ScrollView background transparent
        scrollView.drawsBackground = false
        
        // Set formatting delegate
        DispatchQueue.main.async {
            context.coordinator.parent.formattingDelegate = context.coordinator
        }

        return scrollView
    }

    func updateNSView(_ nsView: NSScrollView, context: Context) {
        guard let textView = nsView.documentView as? CustomInlineNSTextView else { return }
        
        // This logic prevents overwriting NSTextView's content if the change originated from NSTextView itself.
        // 'text' binding should reflect committed text.
        if textView.committedText() != text && !context.coordinator.isInternallyUpdatingTextBinding {
            print("updateNSView: External change to 'text' binding. Updating NSTextView's committed text.")
            textView.clearGhostText() // Clear any existing ghost text
            llmEngine.abortCurrentSuggestion() // Abort LLM
            
            textView.string = text // Set the entire string
            textView.selectedRange = NSRange(location: text.utf16.count, length: 0) // Move cursor to end
            context.coordinator.currentCommittedText = text // Update coordinator's record
        }
        
        if textView.isEditable != true { // Assuming always editable for this demo
            textView.isEditable = true
        }
    }

    // --- Coordinator ---
    class Coordinator: NSObject, NSTextViewDelegate {
        var parent: InlineSuggestingTextView
        var llmEngine: LLMEngine
        weak var managedTextView: CustomInlineNSTextView?

        var currentCommittedText: String = ""
        var isInternallyUpdatingTextBinding: Bool = false
        var isProcessingAcceptOrDismiss: Bool = false

        init(_ parent: InlineSuggestingTextView, llmEngine: LLMEngine) {
            self.parent = parent
            self.llmEngine = llmEngine
            print("Coordinator: Initialized. Initial parent.text = '\(parent.text)'")
            self.currentCommittedText = parent.text
        }

        func textDidChange(_ notification: Notification) {
            guard let textView = notification.object as? CustomInlineNSTextView else {
                print("Coordinator.textDidChange: Notification object is not CustomInlineNSTextView.")
                return
            }
            print("Coordinator.textDidChange: Called.")

            if isProcessingAcceptOrDismiss {
                print("Coordinator.textDidChange: Exiting early, isProcessingAcceptOrDismiss is true.")
                return
            }

            let previousCommittedTextInCoordinator = self.currentCommittedText
            let newCommittedTextFromTextView = textView.committedText()
            let ghostTextBeforeChange = textView.ghostText() ?? "nil"

            print("Coordinator.textDidChange: PrevCommittedText (Coordinator): [\(previousCommittedTextInCoordinator)], NewCommittedText (TextView): [\(newCommittedTextFromTextView)], GhostText before change: [\(ghostTextBeforeChange)]")
            
            if parent.text != newCommittedTextFromTextView {
                print("Coordinator.textDidChange: Updating parent.text from '\(parent.text)' to '\(newCommittedTextFromTextView)'")
                isInternallyUpdatingTextBinding = true
                parent.text = newCommittedTextFromTextView
                DispatchQueue.main.async { self.isInternallyUpdatingTextBinding = false }
            }
            
            if newCommittedTextFromTextView.isEmpty {
                print("Coordinator.textDidChange: Committed text is empty. Clearing ghost text and aborting LLM.")
                textView.clearGhostText()
                llmEngine.abortCurrentSuggestion()
                parent.debouncer.cancel()
                self.currentCommittedText = newCommittedTextFromTextView
                return
            }
            
            let committedTextChangedByUser = newCommittedTextFromTextView != previousCommittedTextInCoordinator
            let ghostTextAfterChange = textView.ghostText() ?? "nil"
            let ghostTextVanished = textView.ghostText() == nil

            print("Coordinator.textDidChange: committedTextChangedByUser: \(committedTextChangedByUser) (New: '\(newCommittedTextFromTextView)' vs Prev: '\(previousCommittedTextInCoordinator)')")
            print("Coordinator.textDidChange: ghostTextVanished: \(ghostTextVanished) (GhostText: '\(ghostTextAfterChange)', isProcessingAcceptOrDismiss: \(isProcessingAcceptOrDismiss))")

            let shouldFetch = committedTextChangedByUser || (ghostTextVanished && !isProcessingAcceptOrDismiss)
            
            if shouldFetch {
                print("Coordinator.textDidChange: Debouncing LLM call. Prompt: '\(newCommittedTextFromTextView)'")
                parent.debouncer.debounce { [weak self, weak textView] in
                    guard let self = self, let textView = textView else { return }
                    
                    let currentPromptForLLM = textView.committedText()
                    if !currentPromptForLLM.isEmpty {
                        print("Coordinator.debouncer: Requesting suggestion for prompt: '\(currentPromptForLLM)'")
                        
                        if let existingGhostText = textView.ghostText() {
                            print("Coordinator.debouncer: Clearing existing ghost text '\(existingGhostText)' before new suggestion.")
                            textView.clearGhostText()
                        }

                        self.llmEngine.generateSuggestion(prompt: currentPromptForLLM) { [weak textView] token in
                            DispatchQueue.main.async {
                                print("Coordinator.debouncer.tokenCallback: Received token '\(token)' for prompt '\(currentPromptForLLM)'")
                                textView?.appendGhostTextToken(token)
                            }
                        } onComplete: { [weak textView] result in
                            DispatchQueue.main.async {
                                guard let textView = textView else { return }
                                switch result {
                                case .success(let fullSuggestion):
                                    print("Coordinator.debouncer.onComplete: Success. Full suggestion: '\(fullSuggestion)' for prompt '\(currentPromptForLLM)'")
                                    if fullSuggestion.isEmpty && textView.ghostText() != nil {
                                        print("Coordinator.debouncer.onComplete: Empty suggestion returned, clearing ghost text.")
                                        textView.clearGhostText()
                                    }
                                case .failure(let error):
                                    print("Coordinator.debouncer.onComplete: LLM suggestion error: \(error) for prompt '\(currentPromptForLLM)'")
                                    if case LLMError.aborted = error {
                                        print("Coordinator.debouncer.onComplete: Suggestion was aborted, not clearing ghost text.")
                                    } else {
                                        print("Coordinator.debouncer.onComplete: Error was not '.aborted', clearing ghost text.")
                                        textView.clearGhostText()
                                    }
                                }
                            }
                        }
                    } else {
                        print("Coordinator.debouncer: Prompt is empty at execution. Clearing ghost text and aborting LLM.")
                        textView.clearGhostText()
                        self.llmEngine.abortCurrentSuggestion()
                    }
                }
            } else {
                print("Coordinator.textDidChange: Not fetching new suggestion. committedTextChangedByUser=\(committedTextChangedByUser), ghostTextVanished=\(ghostTextVanished), isProcessingAcceptOrDismiss=\(isProcessingAcceptOrDismiss)")
            }
            
            self.currentCommittedText = newCommittedTextFromTextView
        }
        
        func textView(_ textView: NSTextView, shouldChangeTextIn affectedCharRange: NSRange, replacementString: String?) -> Bool {
            guard let tv = textView as? CustomInlineNSTextView else { return true }
            
            let currentFullText = tv.string
            let replacement = replacementString ?? ""
            let ghostText = tv.ghostText() ?? "nil"
            print("Coordinator.shouldChangeTextIn: Range: \(affectedCharRange), Replacement: '\(replacement)', CurrentFullText: '\(currentFullText)', GhostText: '\(ghostText)'") // Removed ?? "nil"

            if isProcessingAcceptOrDismiss {
                print("Coordinator.shouldChangeTextIn: Allowing change, isProcessingAcceptOrDismiss is true.")
                return true
            }

            if let ghostRange = tv.currentGhostTextRange {
                print("Coordinator.shouldChangeTextIn: Ghost text exists at \(ghostRange).")
                
                // Case 1: Change overlaps with or is at the start of the ghost text
                if NSMaxRange(affectedCharRange) > ghostRange.location || affectedCharRange.location == ghostRange.location {
                    print("Coordinator.shouldChangeTextIn: Change (range \(affectedCharRange)) overlaps or is at start of ghost text (loc \(ghostRange.location)).")
                    
                    // Scenario: Typing a character that exactly matches the start of the ghost text
                    // Conditions: replacement is not empty, ghost text starts with replacement,
                    //             caret is at the beginning of ghost text (affectedCharRange.location == ghostRange.location),
                    //             and it's an insertion (affectedCharRange.length == 0).
                    if !replacement.isEmpty &&
                       tv.ghostText()?.hasPrefix(replacement) == true &&
                       affectedCharRange.location == ghostRange.location &&
                       affectedCharRange.length == 0 {
                        print("Coordinator.shouldChangeTextIn: Consuming ghost text with '\(replacement)'.")
                        tv.consumeGhostText(length: replacement.utf16.count)
                        
                        // Update committed text in binding and coordinator state immediately after consumption
                        let newCommittedStr = tv.committedText() // This now includes the consumed part
                        if parent.text != newCommittedStr {
                            isInternallyUpdatingTextBinding = true
                            parent.text = newCommittedStr
                            DispatchQueue.main.async { self.isInternallyUpdatingTextBinding = false }
                        }
                        self.currentCommittedText = newCommittedStr
                        
                        parent.debouncer.cancel() // User is confirming part of suggestion.
                                                  // textDidChange will be called. If only ghost text was consumed,
                                                  // newCommittedStr might be same as previousCommittedTextInCoordinator
                                                  // if consumeGhostText doesn't update the underlying storage in a way that
                                                  // changes what committedText() returns *before* textDidChange.
                                                  // This needs to be robust.
                                                  // For now, let's assume consumeGhostText makes the consumed part "real".
                        return false // We handled the event by consuming ghost text.
                    } else {
                        // Any other modification that touches the ghost text area (typing non-matching, deleting, complex selection)
                        print("Coordinator.shouldChangeTextIn: Non-matching char or deletion affecting ghost text. Clearing ghost text.")
                        tv.clearGhostText()
                        llmEngine.abortCurrentSuggestion() // Abort LLM as context is changing significantly
                        parent.debouncer.cancel() // Cancel any pending fetches; textDidChange will trigger a new one.
                        return true // Allow the user's change (e.g., typing the new char, deleting) to proceed on the now non-ghost area.
                    }
                }
                // Case 2: Change is entirely before ghost text (e.g., editing committed text)
                else if NSMaxRange(affectedCharRange) <= ghostRange.location {
                     print("Coordinator.shouldChangeTextIn: Change (range \(affectedCharRange)) is entirely before ghost text (loc \(ghostRange.location)). Clearing ghost text.")
                    tv.clearGhostText()
                    llmEngine.abortCurrentSuggestion() // Abort LLM as context before suggestion changed
                    parent.debouncer.cancel() // Cancel any pending; textDidChange will trigger a new one.
                    return true // Allow the change to proceed.
                }
            } else {
                 print("Coordinator.shouldChangeTextIn: No ghost text present. Allowing change.")
            }
            return true // Allow change by default if no specific ghost text interaction.
        }
        
        // Required for NSTextViewDelegate, but we handle selection changes if needed elsewhere (e.g., for context)
        func textViewDidChangeSelection(_ notification: Notification) {
            // Can be used to clear suggestion if user moves cursor away, etc.
            // For now, let's keep it simple.
        }
    }
}

extension InlineSuggestingTextView.Coordinator: TextFormattingDelegate {
    // Метод для жирного текста
     func toggleBold() {
         guard let textView = managedTextView else { return }
         
         let selectedRange = textView.selectedRange
         if selectedRange.length == 0 {
             // Переключаем состояние для будущего ввода
             if let font = textView.typingAttributes[.font] as? NSFont {
                 var newTraits = font.fontDescriptor.symbolicTraits
                 if newTraits.contains(.bold) {
                     newTraits.remove(.bold) // Убираем жирный стиль
                 } else {
                     newTraits.insert(.bold) // Добавляем жирный стиль
                 }
                 let newFont = NSFont(descriptor: font.fontDescriptor.withSymbolicTraits(newTraits), size: font.pointSize)
                 textView.typingAttributes[.font] = newFont ?? font
             }
             return
         }
         
         textView.textStorage?.enumerateAttribute(.font, in: selectedRange) { value, range, _ in
             if let font = value as? NSFont {
                 var newTraits = font.fontDescriptor.symbolicTraits
                 if newTraits.contains(.bold) {
                     newTraits.remove(.bold) // Убираем жирный стиль
                 } else {
                     newTraits.insert(.bold) // Добавляем жирный стиль
                 }
                 let newFont = NSFont(descriptor: font.fontDescriptor.withSymbolicTraits(newTraits), size: font.pointSize)
                 textView.textStorage?.addAttribute(.font, value: newFont ?? font, range: range)
             }
         }
     }

     // Метод для курсива
     func toggleItalic() {
         guard let textView = managedTextView else { return }
         
         let selectedRange = textView.selectedRange
         if selectedRange.length == 0 {
             // Переключаем состояние для будущего ввода
             if let font = textView.typingAttributes[.font] as? NSFont {
                 var newTraits = font.fontDescriptor.symbolicTraits
                 if newTraits.contains(.italic) {
                     newTraits.remove(.italic) // Убираем курсив
                 } else {
                     newTraits.insert(.italic) // Добавляем курсив
                 }
                 let newFont = NSFont(descriptor: font.fontDescriptor.withSymbolicTraits(newTraits), size: font.pointSize)
                 textView.typingAttributes[.font] = newFont ?? font
             }
             return
         }
         
         textView.textStorage?.enumerateAttribute(.font, in: selectedRange) { value, range, _ in
             if let font = value as? NSFont {
                 var newTraits = font.fontDescriptor.symbolicTraits
                 if newTraits.contains(.italic) {
                     newTraits.remove(.italic) // Убираем курсив
                 } else {
                     newTraits.insert(.italic) // Добавляем курсив
                 }
                 let newFont = NSFont(descriptor: font.fontDescriptor.withSymbolicTraits(newTraits), size: font.pointSize)
                 textView.textStorage?.addAttribute(.font, value: newFont ?? font, range: range)
             }
         }
     }

     // Метод для подчеркивания
     func toggleUnderline() {
         guard let textView = managedTextView else { return }
         
         let selectedRange = textView.selectedRange
         if selectedRange.length == 0 {
             let currentUnderline = textView.typingAttributes[.underlineStyle] as? Int ?? 0
             textView.typingAttributes[.underlineStyle] = currentUnderline == 0 ? NSUnderlineStyle.single.rawValue : 0
             return
         }
         
         textView.textStorage?.enumerateAttribute(.underlineStyle, in: selectedRange) { value, range, _ in
             let currentValue = value as? Int ?? 0
             let newValue = currentValue == 0 ? NSUnderlineStyle.single.rawValue : 0
             textView.textStorage?.addAttribute(.underlineStyle, value: newValue, range: range)
         }
     }

     // Метод для зачеркивания
     func toggleStrikethrough() {
         guard let textView = managedTextView else { return }
         
         let selectedRange = textView.selectedRange
         if selectedRange.length == 0 {
             let currentStrikethrough = textView.typingAttributes[.strikethroughStyle] as? Int ?? 0
             textView.typingAttributes[.strikethroughStyle] = currentStrikethrough == 0 ? NSUnderlineStyle.single.rawValue : 0
             return
         }
         
         textView.textStorage?.enumerateAttribute(.strikethroughStyle, in: selectedRange) { value, range, _ in
             let currentValue = value as? Int ?? 0
             let newValue = currentValue == 0 ? NSUnderlineStyle.single.rawValue : 0
             textView.textStorage?.addAttribute(.strikethroughStyle, value: newValue, range: range)
         }
     }
    
    func toggleBulletList() {
        guard let textView = managedTextView else { return }
        // Простая реализация - добавляем "• " в начало строки
        let selectedRange = textView.selectedRange
        let text = textView.string as NSString
        
        // Найдем начало текущей строки
        let lineRange = text.lineRange(for: selectedRange)
        let line = text.substring(with: lineRange)
        
        if line.hasPrefix("• ") {
            // Удаляем bullet
            let newLine = String(line.dropFirst(2))
            textView.textStorage?.replaceCharacters(in: lineRange, with: newLine)
        } else {
            // Добавляем bullet
            let newLine = "• " + line
            textView.textStorage?.replaceCharacters(in: lineRange, with: newLine)
        }
    }
    
    func toggleNumberedList() {
        guard let textView = managedTextView else { return }
        // Простая реализация - добавляем "1. " в начало строки
        let selectedRange = textView.selectedRange
        let text = textView.string as NSString
        
        let lineRange = text.lineRange(for: selectedRange)
        let line = text.substring(with: lineRange)
        
        // Проверяем, есть ли уже нумерация
        let numberPattern = #"^\d+\.\s+"#
        if line.range(of: numberPattern, options: .regularExpression) != nil {
            // Удаляем нумерацию
            let cleanLine = line.replacingOccurrences(of: numberPattern, with: "", options: .regularExpression)
            textView.textStorage?.replaceCharacters(in: lineRange, with: cleanLine)
        } else {
            // Добавляем нумерацию
            let newLine = "1. " + line
            textView.textStorage?.replaceCharacters(in: lineRange, with: newLine)
        }
    }
    
    func setTextAlignment(_ alignment: NSTextAlignment) {
        guard let textView = managedTextView else { return }
        
        let selectedRange = textView.selectedRange
        if selectedRange.length == 0 {
            // Создаем paragraph style для typing attributes
            let paragraphStyle = NSMutableParagraphStyle()
            paragraphStyle.alignment = alignment
            textView.typingAttributes[.paragraphStyle] = paragraphStyle
            return
        }
        
        // Применяем alignment к выделенному тексту
        textView.textStorage?.enumerateAttribute(.paragraphStyle, in: selectedRange) { value, range, _ in
            let paragraphStyle = (value as? NSParagraphStyle)?.mutableCopy() as? NSMutableParagraphStyle ?? NSMutableParagraphStyle()
            paragraphStyle.alignment = alignment
            textView.textStorage?.addAttribute(.paragraphStyle, value: paragraphStyle, range: range)
        }
    }
    
    func toggleHighlight() {
        guard let textView = managedTextView else { return }
        
        let selectedRange = textView.selectedRange
        
        if selectedRange.length == 0 {
            // Переключаем highlight для будущего ввода
            let currentBackground = textView.typingAttributes[.backgroundColor] as? NSColor
            let isHighlighted = currentBackground == NSColor.yellow
            textView.typingAttributes[.backgroundColor] = isHighlighted ? NSColor.clear : NSColor.yellow
            return
        }
        
        // Применяем highlight к выделенному тексту
        textView.textStorage?.enumerateAttribute(.backgroundColor, in: selectedRange) { value, range, _ in
            let currentBackground = value as? NSColor
            let isHighlighted = currentBackground == NSColor.yellow
            let newBackground = isHighlighted ? NSColor.clear : NSColor.yellow
            textView.textStorage?.addAttribute(.backgroundColor, value: newBackground, range: range)
        }
    }
    
    func increaseFontSize() {
        guard let textView = managedTextView else { return }
        
        let selectedRange = textView.selectedRange
        
        if selectedRange.length == 0 {
            // Увеличиваем размер для будущего ввода
            if let font = textView.typingAttributes[.font] as? NSFont {
                let newSize = min(font.pointSize + 2, 72) // Максимум 72pt
                let newFont = NSFont(descriptor: font.fontDescriptor, size: newSize)
                textView.typingAttributes[.font] = newFont ?? font
            }
            return
        }
        
        // Применяем к выделенному тексту
        textView.textStorage?.enumerateAttribute(.font, in: selectedRange) { value, range, _ in
            if let font = value as? NSFont {
                let newSize = min(font.pointSize + 2, 72)
                let newFont = NSFont(descriptor: font.fontDescriptor, size: newSize)
                textView.textStorage?.addAttribute(.font, value: newFont ?? font, range: range)
            }
        }
    }
    
    func decreaseFontSize() {
        guard let textView = managedTextView else { return }
        
        let selectedRange = textView.selectedRange
        
        if selectedRange.length == 0 {
            // Уменьшаем размер для будущего ввода
            if let font = textView.typingAttributes[.font] as? NSFont {
                let newSize = max(font.pointSize - 2, 8) // Минимум 8pt
                let newFont = NSFont(descriptor: font.fontDescriptor, size: newSize)
                textView.typingAttributes[.font] = newFont ?? font
            }
            return
        }
        
        // Применяем к выделенному тексту
        textView.textStorage?.enumerateAttribute(.font, in: selectedRange) { value, range, _ in
            if let font = value as? NSFont {
                let newSize = max(font.pointSize - 2, 8)
                let newFont = NSFont(descriptor: font.fontDescriptor, size: newSize)
                textView.textStorage?.addAttribute(.font, value: newFont ?? font, range: range)
            }
        }
    }
    
    /// Set the font for the current selection or typing attributes
    func setFont(_ font: NSFont) {
        guard let textView = managedTextView else { return }
        let selectedRange = textView.selectedRange
        if selectedRange.length == 0 {
            textView.typingAttributes[.font] = font
        } else {
            textView.textStorage?.addAttribute(.font, value: font, range: selectedRange)
        }
    }
}

extension InlineSuggestingTextView.Coordinator: LLMInteractionDelegate {
    func acceptSuggestion() {
        print("Coordinator.acceptSuggestion: Called.")
        guard let textView = managedTextView, textView.ghostText() != nil else { return }
        isProcessingAcceptOrDismiss = true
        
        textView.acceptGhostText()
        llmEngine.abortCurrentSuggestion()
        parent.debouncer.cancel()

        // Update parent's text binding
        let newCommittedStr = textView.string
        if parent.text != newCommittedStr {
            isInternallyUpdatingTextBinding = true
            parent.text = newCommittedStr
            DispatchQueue.main.async { self.isInternallyUpdatingTextBinding = false }
        }
        currentCommittedText = newCommittedStr
        
        // Move cursor to end of accepted text
        textView.selectedRange = NSRange(location: newCommittedStr.utf16.count, length: 0)
        textView.scrollRangeToVisible(textView.selectedRange)
        
        // Reset the processing flag immediately to allow normal typing
        isProcessingAcceptOrDismiss = false
        
        // Make sure the text view becomes first responder to continue typing
        DispatchQueue.main.async {
            textView.window?.makeFirstResponder(textView)
        }
    }

    func dismissSuggestion() {
        print("Coordinator.dismissSuggestion: Called.")
        guard let textView = managedTextView, textView.ghostText() != nil else { return }
        isProcessingAcceptOrDismiss = true
        
        textView.clearGhostText()
        llmEngine.abortCurrentSuggestion()
        parent.debouncer.cancel()
        
        // Reset the processing flag immediately
        isProcessingAcceptOrDismiss = false
        
        // Ensure focus remains
        DispatchQueue.main.async {
            textView.window?.makeFirstResponder(textView)
        }
    }
}

// --- Custom NSTextView Subclass ---
class CustomInlineNSTextView: NSTextView {
    weak var llmInteractionDelegate: LLMInteractionDelegate?
    private var customLayoutManager: CustomLayoutManager!
    
    var currentGhostTextRange: NSRange?

    // To track if committed text actually changed by user typing vs. programmatic changes
    private var lastCommittedTextForChangeDetection: String = ""

    override init(frame frameRect: NSRect, textContainer container: NSTextContainer?) {
        // 1. Create TextStorage
        let textStorage = NSTextStorage()
        
        // 2. Create CustomLayoutManager and add it to TextStorage
        customLayoutManager = CustomLayoutManager()
        textStorage.addLayoutManager(customLayoutManager)

        // 3. Create TextContainer and add it to LayoutManager
        let newTextContainer = container ?? NSTextContainer(size: CGSize(width: frameRect.width, height: CGFloat.greatestFiniteMagnitude))
        if container == nil { // Only configure if we created it
            newTextContainer.widthTracksTextView = true // Crucial for wrapping
        }
        customLayoutManager.addTextContainer(newTextContainer)
        
        // Initialize with the new TextKit stack
        super.init(frame: frameRect, textContainer: newTextContainer)
        
        // Standard NSTextView setup
        if container == nil { // Only apply these if we are setting up a default container
            self.minSize = NSSize(width: 0, height: frameRect.height)
            self.maxSize = NSSize(width: CGFloat.greatestFiniteMagnitude, height: CGFloat.greatestFiniteMagnitude)
            self.isVerticallyResizable = true
            self.isHorizontallyResizable = false // Text should wrap
            self.autoresizingMask = [.width] // Resize horizontally with superview
        }
        self.lastCommittedTextForChangeDetection = self.string
    }
    
    convenience override init(frame frameRect: NSRect) {
         // This now correctly calls the designated initializer of this class (CustomInlineNSTextView)
         // which will, in turn, call super.init(frame:textContainer:)
         self.init(frame: frameRect, textContainer: nil) 
    }

    required init?(coder: NSCoder) {
        fatalError("init(coder:) has not been implemented")
    }
    
    override var string: String {
        didSet {
            // When string is set externally, update lastCommittedText for change detection
            if !self.hasGhostText() { // Only if no ghost text, assume it's a full committed text update
                self.lastCommittedTextForChangeDetection = super.string
            }
        }
    }

    func didCommittedTextChangeByUser(newCommittedText: String, previousCommittedText: String) -> Bool {
        // This is a simplified check. A more robust way would be to see if the change
        // was not due to programmatic ghost text append/clear.
        // For now, if committed text differs from what binding had, assume user change.
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
            // Ensure the range is valid before attempting to substring
            if ghostRange.location >= 0 && ghostRange.length >= 0 && NSMaxRange(ghostRange) <= ts.string.utf16.count {
                return (ts.string as NSString).substring(with: ghostRange)
            } else {
                // This case should ideally not be reached if currentGhostTextRange is managed correctly.
                // Consider logging an error or handling it gracefully.
                print("Error: ghostText() called with invalid ghostRange: \(ghostRange) for text length \(ts.string.utf16.count)")
                currentGhostTextRange = nil // Invalidate the range
                return nil
            }
        }
        return nil
    }
    
    func hasGhostText() -> Bool {
        // Also check if the range is valid within the current text length
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
            .foregroundColor: NSColor.gray, // Visually distinguish ghost text
            .font: self.font ?? NSFont.systemFont(ofSize: 16) // Ensure font matches
        ]
        
        let insertionPointForNewSuggestion = self.selectedRange.location // This should be where committed text ends

        // Case 1: Appending to an existing, valid ghost text suggestion.
        // This assumes that if currentGhostTextRange is non-nil, it *should* be at the end of the committed text.
        // The Coordinator's logic in textDidChange should ensure ghost text is cleared if user types elsewhere.
        if let existingRange = currentGhostTextRange,
           // Ensure existingRange is still valid before trying to use NSMaxRange
           existingRange.location <= ts.length && NSMaxRange(existingRange) <= ts.length {
            
            // Defensive check: if the selection point (where user might be typing) is NOT at the start of ghost text,
            // it implies something is inconsistent. For robustness, clear old ghost text and start new.
            if insertionPointForNewSuggestion != existingRange.location {
                print("CustomInlineNSTextView.appendGhostTextToken: Insertion point \(insertionPointForNewSuggestion) is not at start of existing ghost range \(existingRange.location). Clearing old ghost text.")
                // This will clear the old visual ghost text.
                // The actual characters might still be there if not properly managed.
                // For safety, let's ensure we replace the old ghost range if it exists.
                if NSMaxRange(existingRange) <= ts.length { // Check again before replacing
                    ts.replaceCharacters(in: existingRange, with: "")
                }
                // Now, treat as starting a new suggestion at the current insertion point.
                currentGhostTextRange = nil // Fall through to "start new suggestion" logic.
            } else {
                // Append to existing ghost text
                let appendLocation = NSMaxRange(existingRange)
                guard appendLocation <= ts.length else {
                    print("Error: appendGhostTextToken - appendLocation \(appendLocation) out of bounds for text length \(ts.length).")
                    ts.endEditing()
                    return
                }
                ts.insert(NSAttributedString(string: token, attributes: attributes), at: appendLocation)
                currentGhostTextRange = NSRange(location: existingRange.location, length: existingRange.length + token.utf16.count)
                // print("CustomInlineNSTextView.appendGhostTextToken: Appended token '\(token)'. New ghost range: \(currentGhostTextRange!)")
            }
        }
        
        // If, after the above, currentGhostTextRange is still nil, it means we need to start a new suggestion.
        if currentGhostTextRange == nil {
            // Case 2: Starting a new suggestion (currentGhostTextRange was nil or invalidated).
            // The Coordinator should have cleared any old ghost text if this is truly a "new" suggestion context.
            // Insert at the current cursor position.
            if currentGhostTextRange != nil { // It was non-nil but invalid
                print("CustomInlineNSTextView.appendGhostTextToken: currentGhostTextRange was non-nil but became invalid. Resetting.")
            }

            guard insertionPointForNewSuggestion <= ts.length else {
                print("Error: appendGhostTextToken - insertionPointForNewSuggestion \(insertionPointForNewSuggestion) out of bounds for text length \(ts.length).")
                ts.endEditing()
                return
            }
            ts.insert(NSAttributedString(string: token, attributes: attributes), at: insertionPointForNewSuggestion)
            currentGhostTextRange = NSRange(location: insertionPointForNewSuggestion, length: token.utf16.count)
            // print("CustomInlineNSTextView.appendGhostTextToken: Inserted new ghost text '\(token)' at \(insertionPointForNewSuggestion). New ghost range: \(currentGhostTextRange!)")
        }
        ts.endEditing()
        
        // After insertion/append, the caret should remain at the start of the ghost text.
        if let finalGhostRange = currentGhostTextRange {
            // If we just created a new ghost text, the selection should be at its start.
            // If we appended, the selection should also be at the start of the whole ghost text.
            self.selectedRange = NSRange(location: finalGhostRange.location, length: 0)
            self.scrollRangeToVisible(finalGhostRange) // Make sure the ghost text is visible
        }
    }

    func clearGhostText() {
        guard let ts = self.textStorage, let ghostRange = currentGhostTextRange, ghostRange.length > 0 else {
            currentGhostTextRange = nil
            return
        }
        // Ensure range is valid before attempting to modify
        if ghostRange.location <= ts.length && NSMaxRange(ghostRange) <= ts.length {
            ts.beginEditing()
            ts.replaceCharacters(in: ghostRange, with: "")
            ts.endEditing()
        } else {
            print("Warning: clearGhostText called with invalid ghostRange: \(ghostRange) for text length \(ts.length).")
        }
        currentGhostTextRange = nil
    }

    func acceptGhostText() {
        guard let ts = self.textStorage, let ghostRange = currentGhostTextRange, ghostRange.length > 0 else { return }
        
        if ghostRange.location <= ts.length && NSMaxRange(ghostRange) <= ts.length {
            // Remove the .isGhostText attribute and change color to normal
            let normalAttributes: [NSAttributedString.Key: Any] = [
                .foregroundColor: NSColor.textColor, // Or your default text color
                .font: self.font ?? NSFont.systemFont(ofSize: 16) // Ensure font matches
                // Add any other attributes that normal text should have, removing ghost-specific ones
            ]
            ts.beginEditing()
            ts.removeAttribute(.isGhostText, range: ghostRange)
            ts.addAttributes(normalAttributes, range: ghostRange) // This will overwrite existing ones like foregroundColor
            ts.endEditing()
            
            // Update last committed text for change detection, as this is now part of committed text
            self.lastCommittedTextForChangeDetection = ts.string
        } else {
            print("Warning: acceptGhostText called with invalid ghostRange: \(ghostRange)")
        }
        currentGhostTextRange = nil
    }
    
    func consumeGhostText(length: Int) {
        guard let ts = self.textStorage, let ghostRange = currentGhostTextRange, length > 0 else { return }

        if length <= ghostRange.length {
            let consumedRange = NSRange(location: ghostRange.location, length: length)
            let remainingGhostLength = ghostRange.length - length
            
            ts.beginEditing()
            // Convert the consumed part to normal text
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
            
            // Update last committed text for change detection
            self.lastCommittedTextForChangeDetection = self.committedText() // Get the new committed text
        } else {
            // Consuming more than available ghost text, effectively accept all and clear
            acceptGhostText()
        }
    }

    override func keyDown(with event: NSEvent) {
        let modifierFlags = event.modifierFlags
        
        // Handle ghost text shortcuts first
        if hasGhostText() {
            if event.keyCode == kVK_Tab || (event.keyCode == kVK_RightArrow && modifierFlags.contains(.command)) { // Tab or Cmd+RightArrow to accept
                llmInteractionDelegate?.acceptSuggestion()
                return // Event handled
            } else if event.keyCode == kVK_Escape { // Escape to dismiss
                llmInteractionDelegate?.dismissSuggestion()
                return // Event handled
            }
            // Allow Right Arrow to consume one character of ghost text if at the start of it
            else if event.keyCode == kVK_RightArrow && !modifierFlags.contains(.shift) && !modifierFlags.contains(.command) && !modifierFlags.contains(.option) && !modifierFlags.contains(.control) {
                if let ghostRange = currentGhostTextRange, self.selectedRange.location == ghostRange.location {
                    if self.ghostText()?.first != nil { // Check if ghostText is not empty and has a first character
                        // Simulate typing the first character of the ghost text
                        // This will trigger shouldChangeTextIn and consumeGhostText
                        // The coordinator's shouldChangeTextIn should handle this.
                        // We need to make sure the event is passed to the coordinator.
                        // For now, let's directly call consume.
                        self.consumeGhostText(length: 1)
                        // Update the selection to be after the consumed character
                        self.selectedRange = NSRange(location: ghostRange.location + 1, length: 0)
                        
                        // Update binding and coordinator state
                        if let coordinator = self.delegate as? InlineSuggestingTextView.Coordinator {
                            let newCommittedStr = self.committedText()
                            if coordinator.parent.text != newCommittedStr {
                                coordinator.isInternallyUpdatingTextBinding = true
                                coordinator.parent.text = newCommittedStr
                                DispatchQueue.main.async { coordinator.isInternallyUpdatingTextBinding = false }
                            }
                            coordinator.currentCommittedText = newCommittedStr
                        }
                        return // Event handled
                    }
                }
            }
        }
        
        super.keyDown(with: event)
    }
    
    // Prevent direct editing/selection of ghost text (basic)
    override func shouldChangeText(in affectedCharRange: NSRange, replacementString: String?) -> Bool {
        // Запрещаем любые изменения внутри ghost text, кроме Tab/Escape
        if let ghostRange = currentGhostTextRange,
           affectedCharRange.location >= ghostRange.location && NSMaxRange(affectedCharRange) <= NSMaxRange(ghostRange) {
            // Это изменение внутри ghost text. Разрешаем только если это часть Tab/Escape (которые обрабатываются в keyDown)
            // или если это consumeGhostText (который должен вызываться из coordinator's shouldChangeTextIn)
            // Для простоты, если delegate (coordinator) разрешает, то разрешаем.
            // Но coordinator должен быть умным.
        }
        
        // Если изменение влияет на область до ghost text, очищаем ghost text
        if let ghostRange = currentGhostTextRange,
           affectedCharRange.location < ghostRange.location {
            // Это изменение до ghost text. Координатор должен решить, очищать ли ghost text.
        }
        
        // Delegate most of this logic to the Coordinator's implementation
        if let coordinatorDecision = self.delegate?.textView?(self, shouldChangeTextIn: affectedCharRange, replacementString: replacementString) {
            return coordinatorDecision
        }
        
        return super.shouldChangeText(in: affectedCharRange, replacementString: replacementString)
    }
    
    // Override selection methods to prevent ghost text selection
    override func setSelectedRange(_ charRange: NSRange) {
        // Если пытаемся выделить ghost text, корректируем диапазон
        if let ghostRange = currentGhostTextRange {
            let adjustedRange = adjustSelectionRange(charRange, ghostRange: ghostRange)
            super.setSelectedRange(adjustedRange)
        } else {
            super.setSelectedRange(charRange)
        }
    }
    
    // Helper method to adjust selection to avoid ghost text
    private func adjustSelectionRange(_ range: NSRange, ghostRange: NSRange) -> NSRange {
        let rangeEnd = NSMaxRange(range)
        let ghostEnd = NSMaxRange(ghostRange)
        
        // Если выделение начинается внутри ghost text, перемещаем в начало
        if range.location >= ghostRange.location && range.location < ghostEnd {
            // Если это просто клик (length 0), ставим курсор в начало ghost text
            if range.length == 0 {
                return NSRange(location: ghostRange.location, length: 0)
            }
            // Если это выделение, начинающееся в ghost text, ограничиваем его началом ghost text
            // Это предотвратит выделение части ghost text.
            // Более сложное поведение (например, разрешить выделение всего ghost text) потребует доп. логики.
            // Пока что, если выделение начинается в ghost text, оно "схлопывается" до курсора в начале ghost text.
            // Или, если мы хотим запретить выделение *в* ghost text, но разрешить выделение *до* него:
            // return NSRange(location: ghostRange.location, length: 0) // Схлопываем
             return NSRange(location: ghostRange.location, length: 0) // Default: cursor at start of ghost
        }
        
        // Если выделение пересекается с ghost text, обрезаем его
        if range.location < ghostRange.location && rangeEnd > ghostRange.location {
            // Выделение заканчивается на начале ghost text
            return NSRange(location: range.location, length: ghostRange.location - range.location)
        }
        
        return range
    }
    
    // Override mouse events to clear ghost text when clicking elsewhere
    override func mouseDown(with event: NSEvent) {
        let point = convert(event.locationInWindow, from: nil)
        let charIndex = characterIndexForInsertion(at: point)
        
        // Если кликнули не в ghost text область, очищаем ghost text
        if hasGhostText(), let ghostRange = currentGhostTextRange {
            if !(charIndex >= ghostRange.location && charIndex <= NSMaxRange(ghostRange)) {
                // Клик вне ghost text
                llmInteractionDelegate?.dismissSuggestion() // Это вызовет clearGhostText
            }
        }
        
        super.mouseDown(with: event)
    }

    // Helper method to get character index for mouse position
    internal override func characterIndexForInsertion(at point: NSPoint) -> Int {
        guard let layoutManager = self.layoutManager,
              let textContainer = self.textContainer else {
            return 0 // Fallback or handle error
        }
        
        // Ensure layout is up-to-date
        layoutManager.ensureLayout(for: textContainer)
        
        let glyphIndex = layoutManager.glyphIndex(for: point, in: textContainer, fractionOfDistanceThroughGlyph: nil)
        return layoutManager.characterIndexForGlyph(at: glyphIndex)
    }
}

// --- Custom NSLayoutManager Subclass ---
class CustomLayoutManager: NSLayoutManager {
    override func drawGlyphs(forGlyphRange glyphsToShow: NSRange, at origin: NSPoint) {
        // The .foregroundColor attribute set on the NSAttributedString for ghost text
        // should be automatically handled by super.drawGlyphs.
        // If more complex visual effects (e.g., shaders, different background) are needed,
        // this is where you would iterate through attribute runs and draw them differently.
        // For example:
        /*
        self.textStorage?.enumerateAttribute(.isGhostText, in: self.characterRangeForGlyphRange(glyphsToShow, actualGlyphRange: nil), options: []) { value, range, stop in
            if let isGhost = value as? Bool, isGhost {
                // Custom drawing for this range of ghost text
                // For instance, change color or draw a special background
                let ghostGlyphRange = self.glyphRange(forCharacterRange: range, actualCharacterRange: nil)
                
                // Example: Draw with a different color (though attributes are preferred)
                // let currentCtx = NSGraphicsContext.current?.cgContext
                // currentCtx?.setFillColor(NSColor.lightGray.cgColor)
                // super.drawGlyphs(forGlyphRange: ghostGlyphRange, at: origin) // This might not work as expected due to recursion or state
                
                // It's often better to let NSAttributedString attributes handle visual styling.
                // If you need truly custom rendering beyond attributes, this is the place,
                // but it requires careful management of drawing state.
            } else {
                // Draw non-ghost text normally
                // super.drawGlyphs(forGlyphRange: self.glyphRange(forCharacterRange: range, actualCharacterRange: nil), at: origin)
            }
        }
        */
        // For now, relying on NSAttributedString's attributes is simpler.
        super.drawGlyphs(forGlyphRange: glyphsToShow, at: origin)
    }

    // To make ghost text non-selectable or behave differently with mouse interactions,
    // you might need to override methods like:
    // - `setTemporaryAttributes(_:forCharacterRange:)` to prevent selection highlighting.
    // - `glyphIndex(for:in:fractionOfDistanceThroughGlyph:)` to redirect clicks.
    // This can get complex and is beyond a simple color change.
}

// --- Debouncer Utility ---
class Debouncer: ObservableObject { // Made ObservableObject for @State usage if needed, though @Binding works fine for this structure
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

// Add this to your new file or an existing AppKit related file
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

// Make sure LLMError is defined or imported if it's a custom error type
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

// Ensure kVK_Tab, kVK_RightArrow, kVK_Escape are available.
// If not, you might need to import Carbon.HIToolbox or define them.
// For example:
#if canImport(Carbon)
import Carbon.HIToolbox
#else
// Define them if Carbon is not available (e.g., for pure SwiftUI projects not linking Carbon)
// These are common key codes, but for AppKit/Cocoa, they are usually available.
let kVK_Tab: UInt16 = 0x30
let kVK_RightArrow: UInt16 = 0x7C
let kVK_Escape: UInt16 = 0x35
#endif


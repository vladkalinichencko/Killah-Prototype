//
//  ContentView.swift
//  Killah Prototype
//
//  Created by Владислав Калиниченко on 07.05.2025.
//

import SwiftUI
import AppKit
import Combine

extension NSAttributedString.Key {
    static let isGhostText = NSAttributedString.Key("com.example.isGhostText")
}

protocol LLMInteractionDelegate: AnyObject {
    func acceptSuggestion()
    func dismissSuggestion()
}

// Protocol for text formatting communication
protocol TextFormattingDelegate: AnyObject {
    func toggleBold()
    func toggleItalic() 
    func toggleUnderline()
}

struct ContentView: View {
    @State private var text: String = ""
    @StateObject private var llmEngine = LLMEngine()
    @State private var debouncer = Debouncer(delay: 0.5) // Ensure debouncer is a @State property
    @State private var textFormattingDelegate: TextFormattingDelegate?

    var body: some View {
        ZStack(alignment: .top) { // Use ZStack to overlay toolbar
            VStack(alignment: .leading, spacing: 0) {
                InlineSuggestingTextView(
                    text: $text,
                    llmEngine: llmEngine,
                    debouncer: $debouncer,
                    formattingDelegate: $textFormattingDelegate
                )
                .frame(minHeight: 150, idealHeight: 300, maxHeight: .infinity)
                .padding(.top, 80) // Увеличили отступ под увеличенный тулбар
            }
            .padding() // Keep overall padding for the VStack
            .background(Color(NSColor.windowBackgroundColor))
            .edgesIgnoringSafeArea(.top)

            // Floating Toolbar
            FloatingToolbar(formattingDelegate: textFormattingDelegate)
                .padding(.top, 15) // Position the toolbar from the top

        }
        .onAppear {
            print("ContentView appeared. Starting LLM Engine.")
            llmEngine.startEngine()
        }
        .onDisappear {
            print("ContentView disappeared. Stopping LLM Engine.")
            llmEngine.stopEngine()
        }
    }
}

struct FloatingToolbar: View {
    weak var formattingDelegate: TextFormattingDelegate?
    
    var body: some View {
        HStack(spacing: 20) { // Увеличили spacing между кнопками
            Button(action: { 
                formattingDelegate?.toggleBold()
            }) {
                Image(systemName: "bold")
                    .font(.system(size: 16, weight: .medium)) // Увеличили размер иконок
                    .foregroundColor(.primary)
            }
            .buttonStyle(ToolbarButtonStyle())

            Button(action: { 
                formattingDelegate?.toggleItalic()
            }) {
                Image(systemName: "italic")
                    .font(.system(size: 16, weight: .medium))
                    .foregroundColor(.primary)
            }
            .buttonStyle(ToolbarButtonStyle())

            Button(action: { 
                formattingDelegate?.toggleUnderline()
            }) {
                Image(systemName: "underline")
                    .font(.system(size: 16, weight: .medium))
                    .foregroundColor(.primary)
            }
            .buttonStyle(ToolbarButtonStyle())
            
            Spacer() // Pushes controls to one side if needed, or add more controls
            
            Button(action: { /* TODO: Settings action */ }) {
                Image(systemName: "gearshape.fill")
                    .font(.system(size: 16, weight: .medium))
                    .foregroundColor(.primary)
            }
            .buttonStyle(ToolbarButtonStyle())
        }
        .padding(.horizontal, 20) // Увеличили горизонтальные отступы
        .padding(.vertical, 12) // Увеличили вертикальные отступы
        .background(
            VisualEffectView(material: .hudWindow, blendingMode: .withinWindow)
        )
        .cornerRadius(16) // Увеличили радиус скругления
        .shadow(color: Color.black.opacity(0.15), radius: 8, x: 0, y: 4) // Улучшили тень
        .frame(minWidth: 280) // Увеличили минимальную ширину
    }
}

// --- Custom Button Style for Toolbar ---
struct ToolbarButtonStyle: ButtonStyle {
    func makeBody(configuration: Configuration) -> some View {
        configuration.label
            .padding(.horizontal, 12)
            .padding(.vertical, 8)
            .background(
                RoundedRectangle(cornerRadius: 8)
                    .fill(configuration.isPressed ? 
                          Color.primary.opacity(0.15) : 
                          Color.clear)
            )
            .scaleEffect(configuration.isPressed ? 0.95 : 1.0)
            .animation(.easeInOut(duration: 0.1), value: configuration.isPressed)
    }
}

// Helper for NSVisualEffectView in SwiftUI
struct VisualEffectView: NSViewRepresentable {
    var material: NSVisualEffectView.Material
    var blendingMode: NSVisualEffectView.BlendingMode

    func makeNSView(context: Context) -> NSVisualEffectView {
        let view = NSVisualEffectView()
        view.material = material
        view.blendingMode = blendingMode
        view.state = .active // Ensure the effect is active
        return view
    }

    func updateNSView(_ nsView: NSVisualEffectView, context: Context) {
        nsView.material = material
        nsView.blendingMode = blendingMode
    }
}

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
    func toggleBold() {
        guard let textView = managedTextView else { return }
        
        // Получаем текущий выделенный диапазон
        let selectedRange = textView.selectedRange
        
        // Если ничего не выделено, форматируем следующий набираемый текст
        if selectedRange.length == 0 {
            // Переключаем состояние для будущего ввода
            if let font = textView.typingAttributes[.font] as? NSFont {
                let newFont = font.fontDescriptor.symbolicTraits.contains(.bold) ? 
                    NSFont(descriptor: font.fontDescriptor.withSymbolicTraits([]), size: font.pointSize) :
                    NSFont(descriptor: font.fontDescriptor.withSymbolicTraits(.bold), size: font.pointSize)
                textView.typingAttributes[.font] = newFont ?? font
            }
            return
        }
        
        // Применяем форматирование к выделенному тексту
        textView.textStorage?.enumerateAttribute(.font, in: selectedRange) { value, range, _ in
            if let font = value as? NSFont {
                let newFont = font.fontDescriptor.symbolicTraits.contains(.bold) ? 
                    NSFont(descriptor: font.fontDescriptor.withSymbolicTraits([]), size: font.pointSize) :
                    NSFont(descriptor: font.fontDescriptor.withSymbolicTraits(.bold), size: font.pointSize)
                textView.textStorage?.addAttribute(.font, value: newFont ?? font, range: range)
            }
        }
    }
    
    func toggleItalic() {
        guard let textView = managedTextView else { return }
        
        let selectedRange = textView.selectedRange
        
        if selectedRange.length == 0 {
            if let font = textView.typingAttributes[.font] as? NSFont {
                let newFont = font.fontDescriptor.symbolicTraits.contains(.italic) ? 
                    NSFont(descriptor: font.fontDescriptor.withSymbolicTraits([]), size: font.pointSize) :
                    NSFont(descriptor: font.fontDescriptor.withSymbolicTraits(.italic), size: font.pointSize)
                textView.typingAttributes[.font] = newFont ?? font
            }
            return
        }
        
        textView.textStorage?.enumerateAttribute(.font, in: selectedRange) { value, range, _ in
            if let font = value as? NSFont {
                let newFont = font.fontDescriptor.symbolicTraits.contains(.italic) ? 
                    NSFont(descriptor: font.fontDescriptor.withSymbolicTraits([]), size: font.pointSize) :
                    NSFont(descriptor: font.fontDescriptor.withSymbolicTraits(.italic), size: font.pointSize)
                textView.textStorage?.addAttribute(.font, value: newFont ?? font, range: range)
            }
        }
    }
    
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
}

extension InlineSuggestingTextView.Coordinator: LLMInteractionDelegate {
    func acceptSuggestion() {
        print("Coordinator.acceptSuggestion: Called.")
        guard let textView = managedTextView, textView.ghostText() != nil else { return }
        isProcessingAcceptOrDismiss = true
        
        _ = textView.ghostText() ?? ""
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
        textView.scrollRangeToVisible(textView.selectedRange) // Ensure it's visible
        
        DispatchQueue.main.async { self.isProcessingAcceptOrDismiss = false }
    }

    func dismissSuggestion() {
        print("Coordinator.dismissSuggestion: Called.")
        guard let textView = managedTextView, textView.ghostText() != nil else { return }
        isProcessingAcceptOrDismiss = true
        
        textView.clearGhostText()
        llmEngine.abortCurrentSuggestion()
        parent.debouncer.cancel()
        
        // Ensure focus remains
        textView.window?.makeFirstResponder(textView)
        
        DispatchQueue.main.async { self.isProcessingAcceptOrDismiss = false }
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
                print("Warning: ghostText() called with invalid ghostRange: \\(ghostRange) for text length \\(ts.string.utf16.count). Clearing ghost text.")
                // Clear the invalid range to prevent crashes
                // This needs to be done carefully, perhaps by calling clearGhostText on the main thread.
                // For now, just return nil and log. The range will be corrected on next clear/append.
                // clearGhostText() // Be cautious calling modifying methods from here.
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
        
        let insertionPointForNewSuggestion = self.selectedRange.location

        if var existingRange = currentGhostTextRange,
           // Ensure existingRange is still valid before trying to use NSMaxRange
           existingRange.location <= ts.length && NSMaxRange(existingRange) <= ts.length {
            
            // Case 1: Appending to an active suggestion stream.
            // The Coordinator should have cleared ghost text if the context changed significantly.
            // So, if currentGhostTextRange exists, we assume we're appending to it.
            // The insertion point for appending is always the end of the current ghost text.
            let appendLocation = NSMaxRange(existingRange)

            guard appendLocation <= ts.length else {
                ts.endEditing()
                print("CustomInlineNSTextView.appendGhostTextToken Error: Append location \\(appendLocation) for ghost text is out of bounds (text length \\(ts.length)). Clearing ghost text.")
                clearGhostText() // Clear invalid ghost text
                // Attempt to insert as a new suggestion if the original cursor pos is valid
                if insertionPointForNewSuggestion <= ts.length {
                    ts.beginEditing() // Re-begin editing
                    ts.insert(NSAttributedString(string: token, attributes: attributes), at: insertionPointForNewSuggestion)
                    currentGhostTextRange = NSRange(location: insertionPointForNewSuggestion, length: token.utf16.count)
                    print("CustomInlineNSTextView.appendGhostTextToken (Recovery): Inserted new ghost text '\\(token)' at \\(insertionPointForNewSuggestion).")
                    ts.endEditing()
                } else {
                    // Cannot recover, cursor also out of bounds
                     print("CustomInlineNSTextView.appendGhostTextToken (Recovery Failed): Original cursor location \\(insertionPointForNewSuggestion) also out of bounds.")
                }
                // Update selection and scroll after potential recovery
                if let finalGhostRange = currentGhostTextRange {
                    self.selectedRange = NSRange(location: finalGhostRange.location, length: 0)
                    self.scrollRangeToVisible(finalGhostRange)
                } else {
                    self.selectedRange = NSRange(location: min(insertionPointForNewSuggestion, ts.length), length: 0)
                }
                return
            }

            ts.insert(NSAttributedString(string: token, attributes: attributes), at: appendLocation)
            existingRange.length += token.utf16.count
            currentGhostTextRange = existingRange
            // print("CustomInlineNSTextView.appendGhostTextToken: Appended token '\\(token)'. New ghost range: \\(currentGhostTextRange!)")
        } else {
            // Case 2: Starting a new suggestion (currentGhostTextRange was nil or invalid).
            // The Coordinator should have cleared any old ghost text if this is truly a "new" suggestion context.
            // Insert at the current cursor position.
            if currentGhostTextRange != nil { // It was non-nil but invalid
                print("CustomInlineNSTextView.appendGhostTextToken: currentGhostTextRange was non-nil but invalid. Clearing and starting new at cursor.")
                currentGhostTextRange = nil // Explicitly nil it out before creating new.
            }

            guard insertionPointForNewSuggestion <= ts.length else {
                ts.endEditing()
                print("CustomInlineNSTextView.appendGhostTextToken Error: Insertion point \\(insertionPointForNewSuggestion) for new ghost text is out of bounds (text length \\(ts.length)).")
                self.selectedRange = NSRange(location: min(insertionPointForNewSuggestion, ts.length), length: 0)
                return
            }
            ts.insert(NSAttributedString(string: token, attributes: attributes), at: insertionPointForNewSuggestion)
            currentGhostTextRange = NSRange(location: insertionPointForNewSuggestion, length: token.utf16.count)
            // print("CustomInlineNSTextView.appendGhostTextToken: Inserted new ghost text '\\(token)' at \\(insertionPointForNewSuggestion). New ghost range: \\(currentGhostTextRange!)")
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
            print("Warning: Ghost text range was invalid, could not clear.")
        }
        currentGhostTextRange = nil
    }

    func acceptGhostText() {
        guard let ts = self.textStorage, let ghostRange = currentGhostTextRange, ghostRange.length > 0 else { return }
        
        if ghostRange.location <= ts.length && NSMaxRange(ghostRange) <= ts.length {
            ts.beginEditing()
            ts.removeAttribute(.isGhostText, range: ghostRange)
            ts.addAttribute(.foregroundColor, value: self.textColor ?? NSColor.textColor, range: ghostRange)
            // Font should already be correct, but can be re-applied if necessary
            // ts.addAttribute(.font, value: self.font ?? NSFont.systemFont(ofSize: 16), range: ghostRange)
            ts.endEditing()
            self.lastCommittedTextForChangeDetection = ts.string // Update after accept
        }
        currentGhostTextRange = nil
    }
    
    func consumeGhostText(length: Int) {
        guard let ts = self.textStorage, let ghostRange = currentGhostTextRange, length > 0 else { return }

        if length <= ghostRange.length {
            ts.beginEditing()
            let consumedRange = NSRange(location: ghostRange.location, length: length)
            ts.removeAttribute(.isGhostText, range: consumedRange)
            ts.addAttribute(.foregroundColor, value: self.textColor ?? NSColor.textColor, range: consumedRange)
            
            currentGhostTextRange = NSRange(location: ghostRange.location + length, length: ghostRange.length - length)
            if currentGhostTextRange?.length == 0 {
                currentGhostTextRange = nil
            }
            ts.endEditing()
            if currentGhostTextRange == nil { // If fully consumed
                 self.lastCommittedTextForChangeDetection = ts.string
            }
        } else {
            acceptGhostText() // Consuming more than available, accept all
        }
    }

    override func keyDown(with event: NSEvent) {
        // Tab: 48, Esc: 53
        if event.keyCode == 48 && hasGhostText() { // Tab
            llmInteractionDelegate?.acceptSuggestion()
            return // Consume event
        } else if event.keyCode == 53 && hasGhostText() { // Escape
            llmInteractionDelegate?.dismissSuggestion()
            return // Consume event
        }
        super.keyDown(with: event)
    }
    
    // Prevent direct editing/selection of ghost text (basic)
    override func shouldChangeText(in affectedCharRange: NSRange, replacementString: String?) -> Bool {
        // Запрещаем любые изменения внутри ghost text, кроме Tab/Escape
        if let ghostRange = currentGhostTextRange,
           affectedCharRange.location >= ghostRange.location &&
           affectedCharRange.location < NSMaxRange(ghostRange) {
            return false // Блокируем редактирование ghost text
        }
        
        // Если изменение влияет на область до ghost text, очищаем ghost text
        if let ghostRange = currentGhostTextRange,
           NSMaxRange(affectedCharRange) > ghostRange.location {
            DispatchQueue.main.async { [weak self] in
                self?.clearGhostText()
                self?.llmInteractionDelegate?.dismissSuggestion()
            }
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
            return NSRange(location: ghostRange.location, length: 0)
        }
        
        // Если выделение пересекается с ghost text, обрезаем его
        if range.location < ghostRange.location && rangeEnd > ghostRange.location {
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
            if charIndex < ghostRange.location || charIndex >= NSMaxRange(ghostRange) {
                clearGhostText()
                llmInteractionDelegate?.dismissSuggestion()
            }
        }
        
        super.mouseDown(with: event)
    }

    // Helper method to get character index for mouse position
    internal override func characterIndexForInsertion(at point: NSPoint) -> Int {
        guard let layoutManager = self.layoutManager,
              let textContainer = self.textContainer else { return 0 }
        
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
            if value != nil { // It's ghost text
                let glyphRangeForGhost = self.glyphRange(forCharacterRange: range, actualCharacterRange: nil)
                let intersection = NSIntersectionRange(glyphsToShow, glyphRangeForGhost)
                if intersection.length > 0 {
                    // Apply custom drawing for this 'intersection' range of glyphs
                    // E.g., change color, draw background, etc.
                    // NSGraphicsContext.current?.saveGraphicsState()
                    // (Set up custom drawing parameters)
                    // super.drawGlyphs(forGlyphRange: intersection, at: origin) // or custom drawing
                    // NSGraphicsContext.current?.restoreGraphicsState()
                }
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

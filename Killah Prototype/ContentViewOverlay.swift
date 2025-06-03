//
//  ContentView.swift
//  Killah Prototype
//
//  Created by Владислав Калиниченко on 03.05.2025.
//

import SwiftUI
import AppKit

// Extension to get caret rect for a given character index in NSTextView
extension NSTextView {
    /// Returns the caret (insertion point) rect for the given character index, in the view's coordinate system.
    func caretRect(forCharacterIndex index: Int) -> CGRect? {
        guard let layoutManager = self.layoutManager,
              let textContainer = self.textContainer else { return nil }

        let stringLength = (self.string as NSString).length
        let safeIndex = min(max(index, 0), stringLength)

        // Convert character index to glyph index
        let glyphIndex = layoutManager.glyphIndexForCharacter(at: safeIndex)
        // Get the bounding rect for the glyph (caret is a zero-width rect at this position)
        var caretRect = layoutManager.boundingRect(forGlyphRange: NSRange(location: glyphIndex, length: 0), in: textContainer)

        // If the text view is empty or the rect has no height, provide a default height.
        if caretRect.height == 0 {
            caretRect.size.height = self.font?.pointSize ?? NSFont.systemFontSize
        }

        // Adjust by textContainerOrigin to get coordinates relative to the NSTextView's bounds.
        let containerOrigin = self.textContainerOrigin
        return caretRect.offsetBy(dx: containerOrigin.x, dy: containerOrigin.y)
    }
}

class CustomNSTextView: NSTextView {
    /// Callback for accepting or dismissing suggestion
    var onAcceptSuggestion: (() -> Void)?
    var onDismissSuggestion: (() -> Void)?
    override func keyDown(with event: NSEvent) {
        // Detect Tab and Escape via characters to support different keyboard layouts
        if let chars = event.charactersIgnoringModifiers {
            if chars == "\t", let suggestion, !suggestion.isEmpty {
                onAcceptSuggestion?()
                return
            }
            if chars == "\u{1b}", let suggestion, !suggestion.isEmpty {
                onDismissSuggestion?()
                return
            }
        }
        super.keyDown(with: event)
    }
    /// The ghost suggestion string to draw after the caret
    var suggestion: String? {
        didSet {
            // Redraw the view when the suggestion changes.
            self.setNeedsDisplay(self.visibleRect)
        }
    }

    // Add observer properties
    private var selectionDidChangeObserver: Any?
    private var textDidChangeObserverForSuggestion: Any? // Renamed to avoid conflict if NSTextView itself uses this name

    override init(frame frameRect: NSRect, textContainer container: NSTextContainer?) {
        super.init(frame: frameRect, textContainer: container)
        commonInit()
    }

    required init?(coder: NSCoder) {
        super.init(coder: coder)
        commonInit()
    }

    private func commonInit() {
        selectionDidChangeObserver = NotificationCenter.default.addObserver(
            forName: NSTextView.didChangeSelectionNotification,
            object: self,
            queue: .main
        ) { [weak self] _ in
            // Redraw when selection (caret position) changes.
            self?.setNeedsDisplay(self?.visibleRect ?? .zero)
        }
        
        textDidChangeObserverForSuggestion = NotificationCenter.default.addObserver(
            forName: NSText.didChangeNotification, // This is the notification sent by the text storage after changes
            object: self.textStorage, // Observe the textStorage for actual text changes
            queue: .main
        ) { [weak self] _ in
            // Redraw when text content changes.
            self?.setNeedsDisplay(self?.visibleRect ?? .zero)
        }
    }

    deinit {
        if let observer = selectionDidChangeObserver {
            NotificationCenter.default.removeObserver(observer)
        }
        if let observer = textDidChangeObserverForSuggestion {
            NotificationCenter.default.removeObserver(observer)
        }
    }

    override func draw(_ dirtyRect: NSRect) {
        super.draw(dirtyRect)
        guard let suggestionText = self.suggestion, !suggestionText.isEmpty, let font = self.font else { return }
        guard let mainLayoutManager = self.layoutManager, let mainTextContainer = self.textContainer else { return }
        let selectedRange = self.selectedRange()
        guard selectedRange.length == 0 else { return } // Only draw for caret

        guard let caretRect = self.caretRect(forCharacterIndex: selectedRange.location) else { return }

        let suggestionAttributes: [NSAttributedString.Key: Any] = [
            .font: font,
            .foregroundColor: NSColor.gray.withAlphaComponent(0.7)
        ]

        NSGraphicsContext.saveGraphicsState()

        let textContainerOriginX = self.textContainerOrigin.x
        let lineFragmentPadding = mainTextContainer.lineFragmentPadding
        
        // X-coordinate where text glyphs actually start on a line in the main text view
        let contentAreaStartX = textContainerOriginX + lineFragmentPadding

        // Caret's X position relative to the start of the glyph drawing area
        let caretXRelativeToContentAreaStart = caretRect.origin.x - contentAreaStartX
        
        // The total width available for drawing glyphs on a full line.
        // This is the text container's width less line fragment padding from both sides.
        let fullGlyphLineWidth = mainTextContainer.containerSize.width - (2 * lineFragmentPadding)
        
        // For the first line of the suggestion, this width is reduced by the caret's position
        // relative to the start of the glyph drawing area on that line.
        let remainingWidthOnFirstLine = max(0, fullGlyphLineWidth - caretXRelativeToContentAreaStart)

        var firstLineText = ""
        var restOfSuggestionText = ""

        // --- Determine how much text fits on the first line ---
        let fullSuggestionAttrString = NSAttributedString(string: suggestionText, attributes: suggestionAttributes)
        // Use a temporary layout manager and text container restricted to the remaining width on the first line
        let tempLayoutManagerForSizing = NSLayoutManager()
        // Important: The container for sizing the first line part should only have remainingWidthOnFirstLine.
        let tempTextContainerForSizing = NSTextContainer(size: CGSize(width: remainingWidthOnFirstLine, height: CGFloat.greatestFiniteMagnitude))
        tempTextContainerForSizing.lineFragmentPadding = 0 // No extra padding for this temporary container, as it's already sized for glyph area
        
        let tempTextStorageForSizing = NSTextStorage(attributedString: fullSuggestionAttrString)
        tempTextStorageForSizing.addLayoutManager(tempLayoutManagerForSizing)
        tempLayoutManagerForSizing.addTextContainer(tempTextContainerForSizing)
        tempLayoutManagerForSizing.ensureLayout(for: tempTextContainerForSizing)

        // Get the range of glyphs that fit in the constrained width
        let glyphRangeForFirstFragment = tempLayoutManagerForSizing.glyphRange(forBoundingRect: CGRect(x: 0, y: 0, width: remainingWidthOnFirstLine, height: CGFloat.greatestFiniteMagnitude), in: tempTextContainerForSizing)
        
        if glyphRangeForFirstFragment.location == 0 && glyphRangeForFirstFragment.length == 0 && !suggestionText.isEmpty {
            firstLineText = ""
            restOfSuggestionText = suggestionText
        } else {
            let charRangeForFirstFragment = tempLayoutManagerForSizing.characterRange(forGlyphRange: glyphRangeForFirstFragment, actualGlyphRange: nil)
            
            if charRangeForFirstFragment.length < suggestionText.count {
                let potentialFirstLine = (suggestionText as NSString).substring(to: charRangeForFirstFragment.length)
                if let lastSpaceRange = potentialFirstLine.range(of: " ", options: .backwards) {
                    let splitIndex = suggestionText.index(suggestionText.startIndex, offsetBy: potentialFirstLine.distance(from: potentialFirstLine.startIndex, to: lastSpaceRange.upperBound))
                    firstLineText = String(suggestionText[..<splitIndex])
                    restOfSuggestionText = String(suggestionText[splitIndex...])
                } else {
                    firstLineText = potentialFirstLine
                    let splitIndex = suggestionText.index(suggestionText.startIndex, offsetBy: charRangeForFirstFragment.length)
                    restOfSuggestionText = String(suggestionText[splitIndex...])
                }
            } else {
                firstLineText = suggestionText
                restOfSuggestionText = ""
            }
        }
        
        let lineHeight = mainLayoutManager.defaultLineHeight(for: font)

        // --- Draw the first line part ---
        if !firstLineText.isEmpty {
            let firstLineAttrString = NSAttributedString(string: firstLineText, attributes: suggestionAttributes)
            let firstLineStorage = NSTextStorage(attributedString: firstLineAttrString)
            let firstLineLayoutManager = NSLayoutManager()
            let firstLineTextContainer = NSTextContainer(size: CGSize(width: remainingWidthOnFirstLine, height: CGFloat.greatestFiniteMagnitude))
            firstLineTextContainer.lineFragmentPadding = 0 
            firstLineLayoutManager.addTextContainer(firstLineTextContainer)
            firstLineStorage.addLayoutManager(firstLineLayoutManager)
            firstLineLayoutManager.ensureLayout(for: firstLineTextContainer)

            let firstLineGlyphCount = firstLineLayoutManager.numberOfGlyphs
            if firstLineGlyphCount > 0 {
                // Draw first line starting exactly at the caret's position
                let firstLineDrawOrigin = CGPoint(x: caretRect.origin.x, y: caretRect.origin.y)
                firstLineLayoutManager.drawGlyphs(forGlyphRange: NSRange(location: 0, length: firstLineGlyphCount), at: firstLineDrawOrigin)
            }
        }

        // --- Draw the rest of the suggestion on subsequent lines ---
        if !restOfSuggestionText.isEmpty {
            let restAttrString = NSAttributedString(string: restOfSuggestionText, attributes: suggestionAttributes)
            let restStorage = NSTextStorage(attributedString: restAttrString)
            let restLayoutManager = NSLayoutManager()
            // For the rest, use the main container's width and its lineFragmentPadding
            let restTextContainer = NSTextContainer(size: CGSize(width: mainTextContainer.containerSize.width, height: CGFloat.greatestFiniteMagnitude))
            restTextContainer.lineFragmentPadding = lineFragmentPadding // Use the main container's padding
            restLayoutManager.addTextContainer(restTextContainer)
            restStorage.addLayoutManager(restLayoutManager)
            restLayoutManager.ensureLayout(for: restTextContainer)

            let restGlyphCount = restLayoutManager.numberOfGlyphs
            if restGlyphCount > 0 {
                let yOffset = lineHeight 
                let restDrawOrigin = CGPoint(
                    x: self.textContainerOrigin.x // Align with the text container's origin, padding handled by restLayoutManager
                    , y: caretRect.origin.y + yOffset // Position it on the line below the caret's line
                )
                restLayoutManager.drawGlyphs(forGlyphRange: NSRange(location: 0, length: restGlyphCount), at: restDrawOrigin)
            }
        }
        NSGraphicsContext.restoreGraphicsState()
    }
}

struct CustomTextView: NSViewRepresentable {
    @Binding var text: String
    @Binding var fontSize: CGFloat
    @Binding var suggestion: String

    class Coordinator: NSObject, NSTextViewDelegate {
        var parent: CustomTextView
        init(_ parent: CustomTextView) { self.parent = parent }
        func textDidChange(_ notification: Notification) {
            guard let textView = notification.object as? CustomNSTextView else { return }
            parent.text = textView.string
        }
        // Accept suggestion: append to text and clear suggestion
        func acceptSuggestion() {
            guard let textView = parent.getTextView() else { return }
            if let suggestion = textView.suggestion, !suggestion.isEmpty {
                let caret = textView.selectedRange().location
                let nsText = NSMutableString(string: textView.string)
                nsText.insert(suggestion, at: caret)
                textView.string = String(nsText)
                textView.setSelectedRange(NSRange(location: caret + suggestion.count, length: 0))
                parent.text = textView.string
                textView.suggestion = ""
                parent.suggestion = ""
            }
        }
        // Dismiss suggestion: just clear suggestion
        func dismissSuggestion() {
            guard let textView = parent.getTextView() else { return }
            textView.suggestion = ""
            parent.suggestion = ""
        }
    }

    // Helper to get NSTextView from NSScrollView
    func getTextView(from nsView: NSScrollView? = nil) -> CustomNSTextView? {
        let scrollView = nsView ?? (NSApp.keyWindow?.contentView?.subviews.compactMap { $0 as? NSScrollView }.first)
        return scrollView?.documentView as? CustomNSTextView
    }

    func makeCoordinator() -> Coordinator {
        Coordinator(self)
    }

    func makeNSView(context: Context) -> NSScrollView {
        // Set up text storage and container
        let textStorage = NSTextStorage()
        let layoutManager = NSLayoutManager()
        let textContainer = NSTextContainer()
        layoutManager.addTextContainer(textContainer)
        textStorage.addLayoutManager(layoutManager)

        let textView = CustomNSTextView(frame: .zero, textContainer: textContainer)
        textView.delegate = context.coordinator
        // Set up suggestion handlers
        textView.onAcceptSuggestion = { [weak coordinator = context.coordinator] in
            coordinator?.acceptSuggestion()
        }
        textView.onDismissSuggestion = { [weak coordinator = context.coordinator] in
            coordinator?.dismissSuggestion()
        }
        textView.string = text
        textView.isEditable = true
        textView.isSelectable = true
        textView.font = NSFont.systemFont(ofSize: fontSize)
        textView.suggestion = suggestion
        textView.autoresizingMask = [.width]
        textView.isVerticallyResizable = true
        textView.isHorizontallyResizable = false
        textView.textContainerInset = NSSize(width: 4, height: 6)
        textView.textContainer?.widthTracksTextView = true
        textView.textContainer?.heightTracksTextView = false
        textView.minSize = NSSize(width: 0, height: 0)
        textView.maxSize = NSSize(width: CGFloat.greatestFiniteMagnitude, height: CGFloat.greatestFiniteMagnitude)

        let scrollView = NSScrollView(frame: .zero)
        scrollView.hasVerticalScroller = true
        scrollView.hasHorizontalScroller = false
        scrollView.borderType = .noBorder
        scrollView.autohidesScrollers = true
        scrollView.documentView = textView
        scrollView.backgroundColor = .clear
        return scrollView
    }

    func updateNSView(_ nsView: NSScrollView, context: Context) {
        let textView = nsView.documentView as! CustomNSTextView
        if textView.string != text {
            textView.string = text
        }
        if textView.font?.pointSize != fontSize {
            textView.font = NSFont.systemFont(ofSize: fontSize)
        }
        if textView.suggestion != suggestion {
            textView.suggestion = suggestion
        }
    }
}

struct ContentView: View {
    @State private var text = "Hello, world!"
    @State private var fontSize: CGFloat = 16
    @State private var caretRect: CGRect = .zero
    @State public var suggestion: String = "autosuggestion here"

    var body: some View {
        VStack(spacing: 0) {
            CustomTextView(text: $text, fontSize: $fontSize, suggestion: $suggestion)
                .frame(height: 200)
                .background(Color.white)
            Spacer(minLength: 0)
        }
        .padding()

        HStack(spacing: 12) {
            Button("Set Example Text") {
                text = "This is example text set from a button."
            }
            Button("Clear") {
                text = ""
            }
            Button("Append Hello") {
                text += " Hello"
            }
        }
        .padding(.top)

        Divider()

        HStack(spacing: 12) {
            Button("Bold") {
                applyFontTrait(.boldFontMask)
            }
            Button("Italic") {
                applyFontTrait(.italicFontMask)
            }
            Button("Increase Size") {
                fontSize += 2
            }
            Button("Decrease Size") {
                fontSize = max(8, fontSize - 2)
            }
        }
    }

    // Helper to apply font traits to the selected text in the NSTextView
    private func applyFontTrait(_ trait: NSFontTraitMask) {
        // Find the key window's first responder and try to cast to NSTextView
        if let textView = NSApp.keyWindow?.firstResponder as? NSTextView {
            let selectedRange = textView.selectedRange()
            guard selectedRange.length > 0 else { return }
            let currentFont = textView.font ?? NSFont.systemFont(ofSize: fontSize)
            let newFont = NSFontManager.shared.convert(currentFont, toHaveTrait: trait)
            textView.textStorage?.addAttribute(.font, value: newFont, range: selectedRange)
        }
    }
}

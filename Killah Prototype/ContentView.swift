//
//  ContentView.swift
//  Killah Prototype
//
//  Created by Владислав Калиниченко on 07.05.2025.
//

import SwiftUI
import AppKit

// DocumentManager removed: using SwiftUI DocumentGroup and FileDocument

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
    func toggleStrikethrough()
    func isBoldActive() -> Bool
    func isItalicActive() -> Bool
    func isUnderlineActive() -> Bool
    func isStrikethroughActive() -> Bool
    func toggleBulletList()
    func toggleNumberedList()
    func setTextAlignment(_ alignment: NSTextAlignment)
    func setFont(_ font: NSFont)
    func toggleHighlight()
    func increaseFontSize()
    func decreaseFontSize()
}

struct ContentView: View {
    @Binding var document: TextDocument
    @EnvironmentObject var llmEngine: LLMEngine
    @EnvironmentObject var audioEngine: AudioEngine
    @State private var debouncer = Debouncer(delay: 0.5)
    @State private var textFormattingDelegate: TextFormattingDelegate?
    
    @State private var isBoldActive = false
    @State private var isItalicActive = false
    @State private var isUnderlineActive = false
    @State private var isStrikethroughActive = false

    var body: some View {
        ZStack(alignment: .top) {
            // Фон, соответствующий титлбару
            Color.clear
                .background(.regularMaterial)
                .ignoresSafeArea()
            
            InlineSuggestingTextView(
                text: $document.text,
                llmEngine: llmEngine,
                audioEngine: audioEngine,
                debouncer: $debouncer,
                formattingDelegate: $textFormattingDelegate,
                onSelectionChange: updateToolbarStates
            )

            // Floating toolbar with system white background
            FloatingToolbar(
                formattingDelegate: textFormattingDelegate,
                isBoldActive: isBoldActive,
                isItalicActive: isItalicActive,
                isUnderlineActive: isUnderlineActive,
                isStrikethroughActive: isStrikethroughActive
            )
                .zIndex(1)
                .padding(.top, 10)
                .padding(.horizontal, 10) // Add horizontal padding to prevent toolbar from touching window edges
        }
        .onAppear {
            llmEngine.startEngine()
            updateToolbarStates()
        }
    }
    
    func updateToolbarStates() {
        guard let delegate = textFormattingDelegate else { return }
        isBoldActive = delegate.isBoldActive()
        isItalicActive = delegate.isItalicActive()
        isUnderlineActive = delegate.isUnderlineActive()
        isStrikethroughActive = delegate.isStrikethroughActive()
    }
}

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
    @StateObject private var llmEngine = LLMEngine()
    @State private var debouncer = Debouncer(delay: 0.5)
    @State private var textFormattingDelegate: TextFormattingDelegate?

    var body: some View {
        VStack {
            // Floating Toolbar - теперь он вверху
            FloatingToolbar(formattingDelegate: textFormattingDelegate)
                .padding(.top, 10) // Убираем лишнее пространство, если нужно

            Spacer()

            // InlineSuggestingTextView с выравниванием по центру
            InlineSuggestingTextView(
                text: $document.text,
                llmEngine: llmEngine,
                debouncer: $debouncer,
                formattingDelegate: $textFormattingDelegate
            )
            .frame(minHeight: 150, idealHeight: 300, maxHeight: .infinity)
            .padding(.horizontal)
            .background(Color(NSColor.windowBackgroundColor)) // Фон для текста
            .cornerRadius(12)
            
            Spacer()
        }
        .onAppear {
            llmEngine.startEngine()
        }
    }
}



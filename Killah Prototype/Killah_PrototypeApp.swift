//
//  Killah_PrototypeApp.swift
//  Killah Prototype
//
//  Created by Владислав Калиниченко on 03.05.2025.
//

import SwiftUI
import AppKit

@main
struct Killah_PrototypeApp: App {
    @StateObject private var llmEngine = LLMEngine()
    @StateObject private var audioEngine = AudioEngine()

    var body: some Scene {
        DocumentGroup(newDocument: TextDocument()) { file in
            ContentView(document: file.$document)
                .environmentObject(llmEngine)
                .environmentObject(audioEngine)
                .containerBackground(.regularMaterial, for: .window)
                .toolbarBackgroundVisibility(.hidden, for: .windowToolbar)
                .onAppear {
                    if let window = NSApplication.shared.windows.first {
                        window.styleMask.insert(.fullSizeContentView)
                        window.titlebarSeparatorStyle = .none
                        window.isMovableByWindowBackground = true
                        window.backgroundColor = .clear
                        window.isOpaque = false
                        window.hasShadow = true
                        window.titlebarAppearsTransparent = true
                    }
                }
        }
        .windowStyle(.automatic)
        .commands {
            MenuCommands()
        }
    }
}

// Removed AppDelegate; lifecycle handled by DocumentGroup

// Simplified MenuCommands: override only About and Format
struct MenuCommands: Commands {
    var body: some Commands {
        // About menu
        CommandGroup(replacing: .appInfo) {
            Button("About Killah") {
                let alert = NSAlert()
                alert.messageText = "About Killah"
                alert.informativeText = "Killah Text Editor\nVersion 1.0\n\n© 2025 Vladislav Kalinichenko"
                alert.alertStyle = .informational
                alert.addButton(withTitle: "OK")
                alert.runModal()
            }
        }

        // Format menu
        CommandMenu("Format") {
            Button("Bold") { NotificationCenter.default.post(name: NSNotification.Name("ToggleBold"), object: nil) }
                .keyboardShortcut("b", modifiers: .command)
            Button("Italic") { NotificationCenter.default.post(name: NSNotification.Name("ToggleItalic"), object: nil) }
                .keyboardShortcut("i", modifiers: .command)
            Button("Underline") { NotificationCenter.default.post(name: NSNotification.Name("ToggleUnderline"), object: nil) }
                .keyboardShortcut("u", modifiers: .command)
            Divider()
            Menu("Text Size") {
                Button("Increase") { NotificationCenter.default.post(name: NSNotification.Name("IncreaseFontSize"), object: nil) }
                    .keyboardShortcut("+", modifiers: .command)
                Button("Decrease") { NotificationCenter.default.post(name: NSNotification.Name("DecreaseFontSize"), object: nil) }
                    .keyboardShortcut("-", modifiers: .command)
            }
        }
    }
}

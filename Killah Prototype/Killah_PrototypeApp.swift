import SwiftUI
import AppKit
import UniformTypeIdentifiers

@main
struct Killah_PrototypeApp: App {
    @StateObject private var llmEngine: LLMEngine
    @StateObject private var audioEngine: AudioEngine
    @StateObject private var modelManager: ModelManager
    @StateObject private var themeManager: ThemeManager

    @NSApplicationDelegateAdaptor(AppDelegate.self) var appDelegate

    init() {
        let createdModelManager = ModelManager()
        let createdThemeManager = ThemeManager()
        let createdLlmEngine = LLMEngine(modelManager: createdModelManager)
        let createdAudioEngine = AudioEngine(llmEngine: createdLlmEngine)

        _modelManager = StateObject(wrappedValue: createdModelManager)
        _themeManager = StateObject(wrappedValue: createdThemeManager)
        _llmEngine = StateObject(wrappedValue: createdLlmEngine)
        _audioEngine = StateObject(wrappedValue: createdAudioEngine)

        createdModelManager.verifyModels()

        AppDelegate.dependencies = .init(
            llmEngine: createdLlmEngine,
            audioEngine: createdAudioEngine,
            themeManager: createdThemeManager,
            modelManager: createdModelManager
        )
    }

    var body: some Scene {
        DocumentGroup(newDocument: TextDocument()) { file in
            ContentView(document: file.$document)
                .environmentObject(llmEngine)
                .environmentObject(audioEngine)
                .environmentObject(themeManager)
                .environmentObject(modelManager)
                .containerBackground(.regularMaterial, for: .window)
                .toolbarBackgroundVisibility(.hidden, for: .windowToolbar)
                .onAppear {
                    if let window = NSApplication.shared.windows.first {
                        if window.title != "Welcome" {
                            themeManager.applyTheme(to: window)
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
                .onChange(of: themeManager.currentTheme) { _, newTheme in
                    DispatchQueue.main.async {
                        themeManager.applyTheme(to: NSApplication.shared.windows.first)
                    }
                }
        }
        .windowStyle(.automatic)
        .commands {
            MenuCommands()
        }



        Settings {
            SettingsView(modelManager: modelManager)
                .environmentObject(themeManager)
        }
    }
}


// Menu Commands
struct MenuCommands: Commands {
    var body: some Commands {
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

        CommandGroup(replacing: .textFormatting) {
            Button("Bold") {
                FormattingCommands.shared.toggleBold()
            }
            .keyboardShortcut("b", modifiers: [.command])
            Button("Italic") {
                FormattingCommands.shared.toggleItalic()
            }
            .keyboardShortcut("i", modifiers: [.command])
            Button("Underline") {
                FormattingCommands.shared.toggleUnderline()
            }
            .keyboardShortcut("u", modifiers: [.command])
            Button("Strikethrough") {
                FormattingCommands.shared.toggleStrikethrough()
            }
            .keyboardShortcut("x", modifiers: [.command, .shift])
        }
    }
}

import SwiftUI
import AppKit
import UniformTypeIdentifiers

@main
struct Killah_PrototypeApp: App {
    @StateObject private var llmEngine: LLMEngine
    @StateObject private var audioEngine: AudioEngine
    @StateObject private var modelManager: ModelManager
    @StateObject private var themeManager: ThemeManager
    @StateObject private var appState = AppStateManager.shared

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
        #if os(macOS)
        // Welcome окно для macOS - главное окно приложения
        WindowGroup("Welcome") {
            WelcomeView()
                .environmentObject(llmEngine)
                .environmentObject(audioEngine)
                .environmentObject(themeManager)
                .environmentObject(modelManager)
                .environmentObject(appState)
                .containerBackground(.regularMaterial, for: .window)
                .toolbarBackgroundVisibility(.hidden, for: .windowToolbar)
                .onAppear {
                    if let window = NSApplication.shared.windows.first(where: { $0.title == "Welcome" }) {
                        window.title = "Welcome"
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
        .windowStyle(.automatic)
        .windowResizability(.contentSize)
        .defaultSize(width: 800, height: 600)
        
        // DocumentGroup для интеграции с системой - создается только при необходимости
        DocumentGroup(newDocument: TextDocument()) { file in
            ContentView(document: file.$document)
                .environmentObject(llmEngine)
                .environmentObject(audioEngine)
                .environmentObject(themeManager)
                .environmentObject(modelManager)
                .environmentObject(appState)
                .containerBackground(.regularMaterial, for: .window)
                .toolbarBackgroundVisibility(.hidden, for: .windowToolbar)
                .onAppear {
                    if let window = NSApplication.shared.windows.first {
                        window.title = "Untitled"
                        themeManager.applyTheme(to: window)
                        window.styleMask.insert(.fullSizeContentView)
                        window.titlebarSeparatorStyle = .none
                        window.isMovableByWindowBackground = true
                        window.backgroundColor = .clear
                        window.isOpaque = false
                        window.hasShadow = true
                        window.titlebarAppearsTransparent = true
                        window.animationBehavior = .documentWindow
                        window.center()
                    }
                }
        }
        .handlesExternalEvents(matching: Set(arrayLiteral: "document"))
        #else
        // На других платформах используем простую логику
        WindowGroup {
            if appState.showWelcome {
                WelcomeView()
                    .environmentObject(llmEngine)
                    .environmentObject(audioEngine)
                    .environmentObject(themeManager)
                    .environmentObject(modelManager)
                    .environmentObject(appState)
                    .containerBackground(.regularMaterial, for: .window)
                    .toolbarBackgroundVisibility(.hidden, for: .windowToolbar)
                    .onAppear {
                        if let window = NSApplication.shared.windows.first {
                            window.title = "Welcome"
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
            } else {
                ForEach(appState.openDocuments.indices, id: \.self) { index in
                    ContentView(document: Binding(
                        get: { appState.openDocuments[index] },
                        set: { appState.openDocuments[index] = $0 }
                    ))
                    .environmentObject(llmEngine)
                    .environmentObject(audioEngine)
                    .environmentObject(themeManager)
                    .environmentObject(modelManager)
                    .environmentObject(appState)
                    .containerBackground(.regularMaterial, for: .window)
                    .toolbarBackgroundVisibility(.hidden, for: .windowToolbar)
                    .onAppear {
                        if let window = NSApplication.shared.windows.first {
                            window.title = "Untitled"
                            themeManager.applyTheme(to: window)
                            window.styleMask.insert(.fullSizeContentView)
                            window.titlebarSeparatorStyle = .none
                            window.isMovableByWindowBackground = true
                            window.backgroundColor = .clear
                            window.isOpaque = false
                            window.hasShadow = true
                            window.titlebarAppearsTransparent = true
                            window.animationBehavior = .documentWindow
                            window.center()
                        }
                    }
                }
            }
        }
        .windowStyle(.automatic)
        .windowResizability(.contentSize)
        .defaultSize(width: 800, height: 600)
        #endif
        
        Settings {
            SettingsView(modelManager: modelManager)
                .environmentObject(themeManager)
        }
        .commands {
            MenuCommands()
        }
    }
}

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

        CommandGroup(replacing: .sidebar) {
            Button("New") {
                #if os(macOS)
                NSDocumentController.shared.newDocument(nil)
                #else
                AppStateManager.shared.createNewDocument()
                #endif
            }
            .keyboardShortcut("n", modifiers: [.command])
            
            Button("Open...") {
                #if os(macOS)
                NSDocumentController.shared.openDocument(nil)
                #else
                let panel = NSOpenPanel()
                panel.allowedContentTypes = [.plainText, .rtf]
                panel.allowsMultipleSelection = false
                
                if panel.runModal() == .OK, let url = panel.url {
                    AppStateManager.shared.openDocument(from: url)
                }
                #endif
            }
            .keyboardShortcut("o", modifiers: [.command])
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

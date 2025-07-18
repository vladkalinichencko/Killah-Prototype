import SwiftUI
import AppKit
import UniformTypeIdentifiers
import SwiftData

@main
struct Killah_PrototypeApp: App {
    @StateObject private var llmEngine: LLMEngine
    @StateObject private var audioEngine: AudioEngine
    @StateObject private var modelManager: ModelManager
    @StateObject private var themeManager: ThemeManager
    @StateObject private var appState = AppStateManager.shared
    private var modelContainer: ModelContainer

    @NSApplicationDelegateAdaptor(AppDelegate.self) var appDelegate

    init() {
        let schema = Schema([Embedding.self])
        let modelConfiguration = ModelConfiguration(isStoredInMemoryOnly: false)
        do {
            modelContainer = try ModelContainer(for: schema, configurations: modelConfiguration)
            print("✅ ModelContainer initialized with storage: \(modelConfiguration.url.path) from App")
        } catch {
            fatalError("Failed to create ModelContainer: \(error)")
        }
        
        let createdModelManager = ModelManager()
        let createdThemeManager = ThemeManager()
        let createdLlmEngine = LLMEngine(modelManager: createdModelManager, modelContainer: modelContainer)
        let createdAudioEngine = AudioEngine(llmEngine: createdLlmEngine)

        _modelManager = StateObject(wrappedValue: createdModelManager)
        _themeManager = StateObject(wrappedValue: createdThemeManager)
        _llmEngine = StateObject(wrappedValue: createdLlmEngine)
        _audioEngine = StateObject(wrappedValue: createdAudioEngine)
        print("ℹ️ Main context in Killah_PrototypeApp: \(ObjectIdentifier(modelContainer.mainContext))")
        AppDelegate.dependencies = .init(
            llmEngine: createdLlmEngine,
            audioEngine: createdAudioEngine,
            themeManager: createdThemeManager,
            modelManager: createdModelManager,
            modelContainer: modelContainer
        )
    }

    var body: some Scene {
        #if os(macOS)
        // Welcome окно для macOS - открывается программно
        Window("Welcome", id: "welcome") {
            WelcomeView()
                .environment(\.modelContext, modelContainer.mainContext)
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
                        // Enable full-size content and transparent title bar – match document windows
                        window.styleMask.insert(.fullSizeContentView)
                        window.titlebarSeparatorStyle = .none
                        window.titlebarAppearsTransparent = true
                        window.isMovableByWindowBackground = true
                        window.backgroundColor = .clear
                        window.isOpaque = false
                        window.hasShadow = true
                        window.animationBehavior = .documentWindow
                        window.center()
                    }
                }
                .onChange(of: themeManager.currentTheme) { _, _ in
                    themeManager.applyAppTheme()
                }
        }
        .windowStyle(.automatic)
        .windowResizability(.contentSize)
        .defaultSize(width: 800, height: 600)
        
        // DocumentGroup для интеграции с системой - создается только при необходимости
        DocumentGroup(newDocument: TextDocument()) { file in
            ContentView(document: file.$document)
                .environment(\.modelContext, modelContainer.mainContext)
                .environmentObject(llmEngine)
                .environmentObject(audioEngine)
                .environmentObject(themeManager)
                .environmentObject(modelManager)
                .environmentObject(appState)
                .containerBackground(.regularMaterial, for: .window)
                .toolbarBackgroundVisibility(.hidden, for: .windowToolbar)
                .onAppear {
                    // Закрываем Welcome окно при открытии документа
                    if let welcomeWindow = NSApplication.shared.windows.first(where: { $0.title == "Welcome" }) {
                        welcomeWindow.close()
                    }
                    
                    if let window = NSApplication.shared.windows.first {
                        window.title = "Untitled"
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
                .onChange(of: themeManager.currentTheme) { _, _ in
                    themeManager.applyAppTheme()
                }
        }
        .handlesExternalEvents(matching: Set(arrayLiteral: "document"))
        #else
        // На других платформах используем простую логику
        WindowGroup {
            if appState.showWelcome {
                WelcomeView()
                    .environment(\.modelContext, modelContainer.mainContext)
                    .environmentObject(llmEngine)
                    .environmentObject(audioEngine)
                    .environmentObject(themeManager)
                    .environmentObject(modelManager)
                    .environmentObject(appState)
                    .containerBackground(.regularMaterial, for: .window)
                    .toolbarBackgroundVisibility(.hidden, for: .windowToolbar)
                    .preferredColorScheme(themeManager.colorScheme)
            } else {
                ForEach(appState.openDocuments.indices, id: \.self) { index in
                    ContentView(document: Binding(
                        get: { appState.openDocuments[index] },
                        set: { appState.openDocuments[index] = $0 }
                    ))
                    .environment(\.modelContext, modelContainer.mainContext)
                    .environmentObject(llmEngine)
                    .environmentObject(audioEngine)
                    .environmentObject(themeManager)
                    .environmentObject(modelManager)
                    .environmentObject(appState)
                    .containerBackground(.regularMaterial, for: .window)
                    .toolbarBackgroundVisibility(.hidden, for: .windowToolbar)
                    .preferredColorScheme(themeManager.colorScheme)
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
                .onAppear {
                    if let window = NSApplication.shared.windows.first(where: { $0.title == "Settings" }) {
                        window.titlebarSeparatorStyle = .none
                        window.isMovableByWindowBackground = true
                        window.setFrameAutosaveName("SettingsWindow")
                        window.styleMask.insert(.resizable)
                    }
                }
                .onChange(of: themeManager.currentTheme) { _, _ in
                    themeManager.applyAppTheme()
                }
        }
        .defaultSize(width: 400, height: 600)
        .commands {
            MenuCommands()
        }
    }
}

struct MenuCommands: Commands {
    var body: some Commands {
        CommandGroup(replacing: .appInfo) {
            Button("About Killah".localized) {
                let alert = NSAlert()
                alert.messageText = "About Killah".localized
                alert.informativeText = "Killah Text Editor\nVersion 1.0\n\n© 2025 Vladislav Kalinichenko".localized
                alert.alertStyle = .informational
                alert.addButton(withTitle: "OK")
                alert.runModal()
            }
        }

        CommandGroup(replacing: .sidebar) {
            Button("New".localized) {
                #if os(macOS)
                NSDocumentController.shared.newDocument(nil)
                #else
                AppStateManager.shared.createNewDocument()
                #endif
            }
            .keyboardShortcut("n", modifiers: [.command])
            
            Button("Open...".localized) {
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
            Button("Bold".localized) {
                FormattingCommands.shared.toggleBold()
            }
            .keyboardShortcut("b", modifiers: [.command])
            Button("Italic".localized) {
                FormattingCommands.shared.toggleItalic()
            }
            .keyboardShortcut("i", modifiers: [.command])
            Button("Underline".localized) {
                FormattingCommands.shared.toggleUnderline()
            }
            .keyboardShortcut("u", modifiers: [.command])
            Button("Strikethrough".localized) {
                FormattingCommands.shared.toggleStrikethrough()
            }
            .keyboardShortcut("x", modifiers: [.command, .shift])
        }
        
    }
}

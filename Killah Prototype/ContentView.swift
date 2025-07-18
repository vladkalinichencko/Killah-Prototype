//
//  ContentView.swift
//  Killah Prototype
//
//  Created by Владислав Калиниченко on 07.05.2025.
//

import SwiftUI
import AppKit
import SwiftData

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
    func isBulletListActive() -> Bool
    func isNumberedListActive() -> Bool
    func isLeftAlignActive() -> Bool
    func isCenterAlignActive() -> Bool
    func isRightAlignActive() -> Bool
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
    @EnvironmentObject var modelManager: ModelManager
    @State private var debouncer = Debouncer(delay: 1.0)
    @State private var textFormattingDelegate: TextFormattingDelegate?
    
    @State private var isBoldActive = false
    @State private var isItalicActive = false
    @State private var isUnderlineActive = false
    @State private var isStrikethroughActive = false
    @State private var caretCoordinator: CaretUICoordinator?
    @State private var viewUpdater: Bool = false
    @State private var isBulletActive = false
    @State private var isNumberedActive = false
    @State private var isLeftAlignActive   = true
    @State private var isCenterAlignActive = false
    @State private var isRightAlignActive  = false

    @EnvironmentObject var appState: AppStateManager

    var body: some View {
        ZStack {
            // Main editor UI
            editorView
            // Loading indicator overlay
            VStack {
                Spacer()
                LoadingOverlayView()
                    .opacity(appState.isPythonScriptsStarting ? 1 : 0)
                    .offset(y: appState.isPythonScriptsStarting ? 0 : 120)
                    .animation(.easeInOut(duration: 0.35), value: appState.isPythonScriptsStarting)
                    .frame(maxWidth: .infinity)
                    .padding(.bottom, 20)
            }
        }
        .onAppear {
            updateToolbarStates()
        }
        .sheet(isPresented: Binding(
            get: { appState.isModelDownloading },
            set: { _ in }
        )) {
            ModelDownloadView(
                modelManager: modelManager,
                missingFiles: (modelManager.status.missingFiles ?? []),
                isDownloading: modelManager.status.isDownloading,
                downloadProgress: modelManager.status.progress
            )
            .environmentObject(appState)
        }
    }
    
    @ViewBuilder
    private var editorView: some View {
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
                onSelectionChange: updateToolbarStates,
                onCoordinatorChange: { coordinator in
                    DispatchQueue.main.async {
                        self.caretCoordinator = coordinator
                    }
                },
                viewUpdater: $viewUpdater
            )

            // Floating toolbar with system white background
            FloatingToolbar(
                formattingDelegate: textFormattingDelegate,
                isBoldActive: isBoldActive,
                isItalicActive: isItalicActive,
                isUnderlineActive: isUnderlineActive,
                isStrikethroughActive: isStrikethroughActive,
                isBulletActive: isBulletActive,
                isNumberedActive: isNumberedActive,
                isLeftAlignActive: isLeftAlignActive,
                isCenterAlignActive: isCenterAlignActive,
                isRightAlignActive: isRightAlignActive
            )
                .zIndex(1)
                .padding(.top, 20)
                .padding(.horizontal, 10) // Add horizontal padding to prevent toolbar from touching window edges
            
            if let coordinator = caretCoordinator {
                if coordinator.shouldShowOverlay {
                    Color.black.opacity(0.1)
                        .ignoresSafeArea()
                        .zIndex(1)
                        .contentShape(Rectangle())
                        .onTapGesture { }
                }

                Group {
                    CaretOverlayView(coordinator: coordinator) // caret itself
                        .zIndex(0)
                    if modelManager.status == .ready {
                        CaretOverlaysView(coordinator: coordinator) // overlays
                            .zIndex(2)
                    }
                }
                .onChange(of: coordinator.caretPositionInWindow) { _, _ in
                    if coordinator.isExpanded {
                        coordinator.updateUIGroupOffset()
                    }
                }
            }
        }
    }
    
    func updateToolbarStates() {
        guard let delegate = textFormattingDelegate else { return }
        isBoldActive = delegate.isBoldActive()
        isItalicActive = delegate.isItalicActive()
        isUnderlineActive = delegate.isUnderlineActive()
        isStrikethroughActive = delegate.isStrikethroughActive()
        isBulletActive   = delegate.isBulletListActive()
        isNumberedActive = delegate.isNumberedListActive()
        isLeftAlignActive   = delegate.isLeftAlignActive()
        isCenterAlignActive = delegate.isCenterAlignActive()
        isRightAlignActive  = delegate.isRightAlignActive()
    }
}

// Wrapper that observes coordinator and repositions caret automatically
struct CaretOverlayView: View {
    @ObservedObject var coordinator: CaretUICoordinator
    var body: some View {
        SmartCaretView(coordinator: coordinator)
            .position(x: coordinator.caretPositionInWindow.x + coordinator.uiGroupOffsetX,
                      y: coordinator.caretPositionInWindow.y - coordinator.caretVerticalOffset)
            .animation(Animation.spring(response: 0.3, dampingFraction: 0.8, blendDuration: 0.1),
                       value: coordinator.caretPositionInWindow)
            .animation(Animation.spring(response: 0.3, dampingFraction: 0.8, blendDuration: 0.1),
                       value: coordinator.caretVerticalOffset)
            .animation(.spring(response: 0.4, dampingFraction: 0.8), value: coordinator.uiGroupOffsetX)
    }
}

// Struct to observe coordinator and update UI group offset
struct CaretOverlaysView: View {
    @ObservedObject var coordinator: CaretUICoordinator
    var body: some View {
        ZStack {
            CaretRecordButton(coordinator: coordinator)
                .offset(x: -(coordinator.caretButtonPadding + 5))

            CaretPromptField(coordinator: coordinator)
                .offset(x: coordinator.caretButtonPadding + coordinator.basePromptFieldWidth/2)

            CaretPauseButton(coordinator: coordinator)
                .offset(y: -coordinator.caretButtonPadding)
            
            CaretStopButton(coordinator: coordinator)
                .offset(y: coordinator.caretButtonPadding + 15)

            AudioWaveformView(coordinator: coordinator)
                .offset(x: coordinator.caretButtonPadding + (120 / 2))
            
            TranscriptionView(coordinator: coordinator)
                .offset(x: -coordinator.caretButtonPadding - (220 / 2))
        }
        .position(x: coordinator.caretPositionInWindow.x, y: coordinator.caretPositionInWindow.y - coordinator.caretVerticalOffset)
        .offset(x: coordinator.uiGroupOffsetX)
        .animation(Animation.spring(response: 0.3, dampingFraction: 0.8, blendDuration: 0.1), value: coordinator.caretPositionInWindow)
        .animation(Animation.spring(response: 0.3, dampingFraction: 0.8, blendDuration: 0.1), value: coordinator.caretVerticalOffset)
        .animation(.spring(response: 0.4, dampingFraction: 0.8), value: coordinator.uiGroupOffsetX)
    }
}

struct ContentView_Previews: PreviewProvider {
    // Создаём зависимости для предварительного просмотра
        private static var previewDependencies: (modelContainer: ModelContainer, modelManager: ModelManager, llmEngine: LLMEngine, audioEngine: AudioEngine) {
            // Определяем схему (замените на актуальные модели)
            let schema = Schema([
                Embedding.self // Указываем модель Embedding, так как она используется в LLMEngine
                // Если есть другие модели, например DocumentItem, добавьте их сюда
            ])
            let config = ModelConfiguration(isStoredInMemoryOnly: false)
            let modelContainer: ModelContainer
            do {
                modelContainer = try ModelContainer(for: schema, configurations: config)
                print("✅ ModelContainer initialized with storage: \(config.url.path) from ContentView")

            } catch {
                fatalError("Не удалось создать ModelContainer для предварительного просмотра: \(error)")
            }

            let modelManager = ModelManager()
            let llmEngine = LLMEngine(modelManager: modelManager, modelContainer: modelContainer)
            let audioEngine = AudioEngine(llmEngine: llmEngine)
            
            return (modelContainer, modelManager, llmEngine, audioEngine)
        }
        
        static var previews: some View {
            let dependencies = previewDependencies
            
            ContentView(
                document: .constant(TextDocument())
            )
            .environmentObject(dependencies.llmEngine)
            .environmentObject(dependencies.audioEngine)
            .environmentObject(dependencies.modelManager)
            .environment(\.modelContext, dependencies.modelContainer.mainContext)
        }
}



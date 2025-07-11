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

    var body: some View {
        ZStack {
            // Main editor UI
            editorView
            // Loading indicator overlay
            VStack {
                Spacer()
                LoadingOverlayView()
                    .opacity(isEngineStarting ? 1 : 0)
                    .offset(y: isEngineStarting ? 0 : 120)
                    .animation(.easeInOut(duration: 0.35), value: isEngineStarting)
                    .frame(maxWidth: .infinity)
                    .padding(.bottom, 20)
            }
        }
        .onAppear {
            updateToolbarStates()
            print("🖥️ ContentView.onAppear — model status = \(modelManager.status)")
            if case .needsDownloading = modelManager.status {
                // Статус уже говорит, что моделей нет – сразу открываем диалог
                showModelDownloadSheet = true
            } else if modelManager.status == .ready {
                llmEngine.startEngine(for: "autocomplete")
                llmEngine.startEngine(for: "audio")
            }
        }
        .task {
            // Проверяем наличие моделей вне фазы построения вью, чтобы избежать предупреждения SwiftUI
            modelManager.verifyModels()
        }
        .onChange(of: modelManager.status) { _, newStatus in
            print("🔄 onChange status -> \(newStatus)")
            switch newStatus {
            case .ready:
                llmEngine.startEngine(for: "autocomplete")
                llmEngine.startEngine(for: "audio")
            case .needsDownloading:
                showModelDownloadSheet = true
                DispatchQueue.main.async {
                    llmEngine.stopEngine()
                }
            default:
                break
            }
        }
        .sheet(isPresented: $showModelDownloadSheet) {
            ModelDownloadView(
                modelManager: modelManager,
                missingFiles: (modelManager.status.missingFiles ?? []),
                isDownloading: modelManager.status.isDownloading,
                downloadProgress: modelManager.status.progress
            )
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
                caretOverlays(coordinator: coordinator)
            }
        }
        .onAppear {
            llmEngine.startEngine(for: "autocomplete")
            llmEngine.startEngine(for: "audio")
            updateToolbarStates()
        }
    }

    @ViewBuilder
    private func caretOverlays(coordinator: CaretUICoordinator) -> some View {
        let verticalAdjustment: CGFloat = 5
        let baseY = coordinator.caretPositionInWindow.y - coordinator.caretSize.height - verticalAdjustment
        let baseX = coordinator.caretPositionInWindow.x - 5
        
        ZStack {
            if coordinator.shouldShowOverlay {
                Color.black.opacity(0.1)
                    .edgesIgnoringSafeArea(.all)
                    .contentShape(Rectangle())
                    .onTapGesture { }
            }
            
            SmartCaretView(coordinator: coordinator)
                .position(x: coordinator.caretPositionInWindow.x, y: baseY)
                .zIndex(0)
            
            CaretRecordButton(coordinator: coordinator)
                .position(x: baseX - coordinator.caretButtonPadding, y: baseY)
                .zIndex(2)
            
            CaretPromptField(coordinator: coordinator)
                .position(x: baseX + coordinator.caretButtonPadding + coordinator.basePromptFieldWidth / 2, y: baseY)
                .zIndex(2)
            
            CaretPauseButton(coordinator: coordinator)
                .position(x: coordinator.caretPositionInWindow.x,
                          y: baseY - coordinator.caretButtonPadding)
                .zIndex(2)
            
            CaretStopButton(coordinator: coordinator)
                .position(x: coordinator.caretPositionInWindow.x,
                          y: baseY + coordinator.caretButtonPadding + 15)
                .zIndex(2)
            
            AudioWaveformView(coordinator: coordinator)
                .position(x: baseX + 8 + 150 / 2, y: baseY)
                .zIndex(2)
            
            TranscriptionView(coordinator: coordinator)
                .position(x: baseX - coordinator.caretButtonPadding - 200 / 2, y: baseY)
                .zIndex(2)
        }
        .animation(Animation.spring(response: 0.3, dampingFraction: 0.8, blendDuration: 0.1), value: coordinator.caretPositionInWindow)
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

    // MARK: - Helper
    private var isEngineStarting: Bool {
        (llmEngine.getRunnerState(for: "autocomplete") == .starting) ||
        (llmEngine.getRunnerState(for: "audio") == .starting)
    }
}

struct ContentView_Previews: PreviewProvider {
    static var previews: some View {
        let modelManager = ModelManager()
        let llmEngine = LLMEngine(modelManager: modelManager)
        let audioEngine = AudioEngine(llmEngine: llmEngine)
        
        ContentView(
            document: .constant(TextDocument())
        )
        .environmentObject(llmEngine)
        .environmentObject(audioEngine)
        .environmentObject(modelManager)
    }
}

extension ModelManager.ModelStatus {
    var isDownloading: Bool {
        if case .downloading = self { return true }
        return false
    }
    
    var progress: Double {
        if case .downloading(let progress) = self { return progress }
        return 0
    }
    
    var missingFiles: [ModelManager.ModelFile]? {
        if case .needsDownloading(let files) = self { return files }
        return nil
    }
}

import SwiftUI
import AppKit

extension View {
    func onPressGesture(onPress: @escaping () -> Void, onRelease: @escaping () -> Void) -> some View {
        self.simultaneousGesture(
            DragGesture(minimumDistance: 0)
                .onChanged { _ in
                    onPress()
                }
                .onEnded { _ in
                    onRelease()
                }
        )
    }
}

struct PressableButtonStyle: ButtonStyle {
    func makeBody(configuration: Configuration) -> some View {
        configuration.label
            .scaleEffect(configuration.isPressed ? 0.9 : 1.0)
            .animation(.easeInOut(duration: 0.1), value: configuration.isPressed)
    }
}

struct SmartCaretView: View {
    @ObservedObject var coordinator: CaretUICoordinator
    @EnvironmentObject var llmEngine: LLMEngine
    
    // Local UI state
    @State private var isCaretHovered: Bool = false
    @State private var isCaretPressed: Bool = false
    @State private var caretXOffset: CGFloat = 0
    
    // Unified animation
    private let caretUIAnimation = Animation.spring(response: 0.3, dampingFraction: 0.8)

    var body: some View {
        ZStack {
            // 1. Visual Caret (not interactive)
            Rectangle()
                .fill(caretColor)
                .frame(
                    width: caretWidth,
                    height: coordinator.caretSize.height
                )
                .scaleEffect(isCaretPressed ? 0.9 : 1.0)
                .shadow(color: isCaretPressed ? Color.red.opacity(0.5) : .clear, radius: 5)
                .shadow(color: hoverGlowColor, radius: 4)
                .shadow(color: generationGlowColor, radius: 4, x: 6, y: 0)
                .shadow(color: generationGlowColor, radius: 8, x: 16, y: 0)
                .shadow(color: generationGlowColor, radius: 4, x: -4, y: 0)
                .offset(x: caretXOffset)
                .animation(caretUIAnimation, value: coordinator.isGenerating)
                .animation(caretUIAnimation, value: isCaretHovered)
                .animation(caretUIAnimation, value: isCaretPressed)
                .onChange(of: coordinator.triggerBounceRight) { _, newValue in
                    if newValue {
                        caretBounceRight()
                        coordinator.triggerBounceRight = false
                    }
                }
                .onChange(of: coordinator.triggerBounceLeft) { _, newValue in
                    if newValue {
                        caretBounceLeft()
                        coordinator.triggerBounceLeft = false
                    }
                }
                .allowsHitTesting(false)

            // 2. Invisible Interaction Area
            Rectangle()
                .fill(Color.primary.opacity(0.0001))
                .frame(
                    width: (coordinator.caretSize.width * 1.4) + 10, // Max visual width + padding
                    height: coordinator.caretSize.height + 10
                )
                .onHover { hovering in
                    withAnimation(caretUIAnimation) {
                        isCaretHovered = hovering
                    }
                }
                .gesture(
                    DragGesture(minimumDistance: 0)
                        .onChanged { _ in
                            // This is the "onPress" part
                            withAnimation(caretUIAnimation) {
                                isCaretPressed = true
                            }
                        }
                        .onEnded { _ in
                            // This is the "onRelease" part
                            withAnimation(caretUIAnimation) {
                                isCaretPressed = false
                            }
                            // AND this is where the action should happen
                            coordinator.toggleExpanded()
                        }
                )
        }
    }
    
    // Local computed properties
    private var caretColor: Color {
        if coordinator.isGenerating {
            return Color.red
        } else if coordinator.isRecording {
            return Color.red.opacity(0.7)
        } else if isCaretHovered {
            return Color.red
        } else {
            return Color.primary
        }
    }
    
    private var generationGlowColor: Color {
        coordinator.isGenerating ? Color.red.opacity(0.7) : .clear
    }
    
    private var hoverGlowColor: Color {
        (!coordinator.isGenerating && isCaretHovered && !isCaretPressed) ? Color.red.opacity(0.4) : .clear
    }
    
    private var caretWidth: CGFloat {
        if isCaretPressed {
            return coordinator.caretSize.width * 1.2
        } else if isCaretHovered {
            return coordinator.caretSize.width * 1.4
        } else {
            return coordinator.caretSize.width
        }
    }
    
    // Animation effects
    // --- Bounce Right ---
    func caretBounceRight() {
        withAnimation(.easeOut(duration: 0.12)) {
            caretXOffset = 5
        }
        DispatchQueue.main.asyncAfter(deadline: .now() + 0.12) {
            withAnimation(.spring(response: 0.4, dampingFraction: 0.6)) {
                caretXOffset = 0
            }
        }
    }

    // --- Bounce Left ---
    func caretBounceLeft() {
        withAnimation(.easeOut(duration: 0.12)) {
            caretXOffset = -5
        }
        DispatchQueue.main.asyncAfter(deadline: .now() + 0.12) {
            withAnimation(.spring(response: 0.4, dampingFraction: 0.6)) {
                caretXOffset = 0
            }
        }
    }
}

struct CaretRecordButton: View {
    @ObservedObject var coordinator: CaretUICoordinator
    
    // Unified animation
    private let caretUIAnimation = Animation.spring(response: 0.3, dampingFraction: 0.8, blendDuration: 0.1)
    
    var body: some View {
        Button(action: {
            if coordinator.isRecording {
                coordinator.stopRecording()
            } else {
                coordinator.startRecording()
            }
        }) {
            ZStack {
                RoundedRectangle(cornerRadius: 8)
                    .fill(.ultraThinMaterial)
                    .overlay(
                        RoundedRectangle(cornerRadius: 8)
                            .stroke(Color.primary.opacity(0.15), lineWidth: 0.75)
                    )

                Circle()
                    .fill(Color.red)
                    .frame(
                        width: coordinator.menuItemSize * 0.6,
                        height: coordinator.menuItemSize * 0.6
                    )
                    .shadow(color: Color.red.opacity(0.4), radius: 4, x: 0, y: 2)
            }
            .frame(
                width: coordinator.menuItemSize + 12,
                height: coordinator.menuItemSize + 12
            )
        }
        .buttonStyle(PressableButtonStyle())
        .shadow(color: Color.black.opacity(0.1), radius: 10, x: 0, y: 3)
        .scaleEffect(x: shouldShowRecordButton ? 1.0 : 0.1, anchor: UnitPoint.trailing)
        .offset(x: shouldShowRecordButton ? 0 : 20)
        .opacity(shouldShowRecordButton ? 1.0 : 0.0)
        .animation(caretUIAnimation, value: coordinator.isExpanded)
        .animation(caretUIAnimation, value: coordinator.isRecording)
        .allowsHitTesting(shouldShowRecordButton)
    }
    
    private var shouldShowRecordButton: Bool {
        return coordinator.isExpanded && !coordinator.isRecording
    }
}

struct CaretPauseButton: View {
    @ObservedObject var coordinator: CaretUICoordinator
    
    // Unified animation  
    private let caretUIAnimation = Animation.spring(response: 0.3, dampingFraction: 0.8, blendDuration: 0.1)
    
    var body: some View {
        Button(action: {
            coordinator.togglePause()
        }) {
            ZStack {
                RoundedRectangle(cornerRadius: 8)
                    .fill(.ultraThinMaterial)
                    .overlay(
                        RoundedRectangle(cornerRadius: 8)
                            .stroke(Color.primary.opacity(0.15), lineWidth: 0.75)
                    )

                Image(systemName: coordinator.isPaused ? "play.fill" : "pause.fill")
                    .font(.system(size: coordinator.menuItemSize * 0.5))
                    .foregroundColor(.red)
                    .shadow(color: Color.red.opacity(0.4), radius: 4, x: 0, y: 2)
            }
            .frame(
                width: coordinator.menuItemSize + 12,
                height: coordinator.menuItemSize + 12
            )
        }
        .buttonStyle(PressableButtonStyle())
        .shadow(color: Color.black.opacity(0.1), radius: 10, x: 0, y: 3)
        .scaleEffect(y: coordinator.isRecording ? 1.0 : 0.1, anchor: UnitPoint.bottom)
        .offset(y: coordinator.isRecording ? -15 : -20)
        .opacity(coordinator.isRecording ? 1.0 : 0.0)
        .animation(caretUIAnimation, value: coordinator.isRecording)
        .allowsHitTesting(coordinator.isRecording)
    }
}

struct CaretStopButton: View {
    @ObservedObject var coordinator: CaretUICoordinator
    
    // Unified animation
    private let caretUIAnimation = Animation.spring(response: 0.3, dampingFraction: 0.8, blendDuration: 0.1)
    
    var body: some View {
        Button(action: {
            coordinator.stopRecording()
        }) {
            ZStack {
                RoundedRectangle(cornerRadius: 8)
                    .fill(.ultraThinMaterial)
                    .overlay(
                        RoundedRectangle(cornerRadius: 8)
                            .stroke(Color.primary.opacity(0.15), lineWidth: 0.75)
                    )

                Image(systemName: "stop.fill")
                    .font(.system(size: coordinator.menuItemSize * 0.5))
                    .foregroundColor(.red)
                    .shadow(color: Color.red.opacity(0.4), radius: 4, x: 0, y: 2)
            }
            .frame(
                width: coordinator.menuItemSize + 12,
                height: coordinator.menuItemSize + 12
            )
        }
        .buttonStyle(PressableButtonStyle())
        .shadow(color: Color.black.opacity(0.1), radius: 10, x: 0, y: 3)
        .scaleEffect(y: coordinator.isRecording ? 1.0 : 0.1, anchor: UnitPoint.top)
        .offset(y: coordinator.isRecording ? 0 : -20)
        .opacity(coordinator.isRecording ? 1.0 : 0.0)
        .animation(caretUIAnimation, value: coordinator.isRecording)
        .allowsHitTesting(coordinator.isRecording)
    }
}

struct CaretPromptField: View {
    @ObservedObject var coordinator: CaretUICoordinator
    
    // Local UI state
    @State private var isPromptFieldHovered: Bool = false
    @State private var allowHoverEffect: Bool = false
    
    // Unified animation
    private let caretUIAnimation = Animation.spring(response: 0.3, dampingFraction: 0.8, blendDuration: 0.1)
    var body: some View {
        TextField("Ask lil Pushkin", text: $coordinator.promptText, axis: .vertical)
            .onSubmit {
                if let textView = (coordinator.textInsertionHandler).self as? NSTextView {
                    let selectedRange = textView.selectedRange()
                    let selectedText = selectedRange.length > 0 ? (textView.string as NSString).substring(with: selectedRange) : nil
                    coordinator.generateFromTextPrompt(selectedText: selectedText)
                } else {
                    coordinator.generateFromTextPrompt()
                }
            }
            .font(.system(size: coordinator.promptFieldFontSize))
            .textFieldStyle(PlainTextFieldStyle())
            .frame(
                width: promptFieldWidth,
                height: coordinator.calculatePromptFieldHeight()
            )
            .padding(.horizontal, 12)
            .padding(.vertical, 6)
            .background(
                RoundedRectangle(cornerRadius: 8)
                    .fill(.ultraThinMaterial)
                    .overlay(
                        RoundedRectangle(cornerRadius: 8)
                            .stroke(Color.primary.opacity(0.15), lineWidth: 0.75)
                    )
            )
            .clipShape(RoundedRectangle(cornerRadius: 8))
            .contentShape(RoundedRectangle(cornerRadius: 8))
            .shadow(color: Color.black.opacity(0.1), radius: 10, x: 0, y: 3)
            .scaleEffect(x: shouldShowPromptField ? 1.0 : 0.1, anchor: .leading)
            .offset(x: shouldShowPromptField ? promptFieldExpansionOffset : -20)
            .opacity(shouldShowPromptField ? 1.0 : 0.0)
            .onHover { hovering in
                if allowHoverEffect {
                    withAnimation(caretUIAnimation) {
                        isPromptFieldHovered = hovering
                    }
                }
            }
            .onChange(of: shouldShowPromptField) { _, isShowing in
                if isShowing {
                    DispatchQueue.main.asyncAfter(deadline: .now() + 0.1) {
                        allowHoverEffect = true
                    }
                } else {
                    allowHoverEffect = false
                    isPromptFieldHovered = false
                }
            }
            .animation(caretUIAnimation, value: coordinator.isExpanded)
            .animation(caretUIAnimation, value: coordinator.isRecording)
            .animation(caretUIAnimation, value: coordinator.promptText)
            .animation(caretUIAnimation, value: promptFieldWidth)
            .allowsHitTesting(shouldShowPromptField && coordinator.isExpanded && !coordinator.isRecording)
    }
    
    // Local computed properties
    private var shouldShowPromptField: Bool {
        return coordinator.isExpanded && !coordinator.isRecording
    }
    
    private var promptFieldWidth: CGFloat {
        if shouldShowPromptField {
            // Expand when hovered OR when there's text in the field
            let expand = isPromptFieldHovered || !coordinator.promptText.isEmpty
            return expand ? coordinator.expandedPromptFieldWidth : coordinator.basePromptFieldWidth
        } else {
            return coordinator.basePromptFieldWidth
        }
    }
    
    private var promptFieldExpansionOffset: CGFloat {
        let expansion = promptFieldWidth - coordinator.basePromptFieldWidth
        return expansion / 2
    }
}

struct AudioWaveformView: View {
    @ObservedObject var coordinator: CaretUICoordinator
    @State private var audioSamples: [Float] = Array(repeating: 0, count: 50)
    @State private var opacity: Double = 0
    @State private var inactivityTimer: Timer?
    @Environment(\.colorScheme) private var colorScheme

    private let timer = Timer.publish(every: 0.02, on: .main, in: .common).autoconnect()
    
    private let caretUIAnimation = Animation.spring(response: 0.3, dampingFraction: 0.8, blendDuration: 0.1)

    private var waveformColors: Color {
        colorScheme == .dark ? .white : .black
    }

    var body: some View {
        ZStack(alignment: .center) {
            Rectangle()
                .fill(.ultraThinMaterial)
                .blur(radius: 8)
                .frame(
                    width: 130,
                    height: coordinator.menuItemSize
                )
                .allowsHitTesting(false)

            waveformContent
                .shadow(color: Color.black, radius: 10, x: 0, y: 0)
        }
            .mask(
                LinearGradient(
                    gradient: Gradient(stops: [
                        .init(color: .clear, location: 0.0),
                        .init(color: .black.opacity(0.0), location: 0.1),
                        .init(color: .black.opacity(0.2), location: 0.4),
                        .init(color: .black.opacity(1.0), location: 0.8),
                        .init(color: .black, location: 1.0)
                    ]),
                    startPoint: .trailing,
                    endPoint: .leading
                )
            )
            .opacity(opacity)
            .onReceive(timer) { _ in
                guard coordinator.isRecording && !coordinator.isPaused else { return }
                audioSamples.removeFirst()
                let newSample = coordinator.audioLevel // * Float.random(in: 0.8...1.2)
                audioSamples.append(newSample)
            }
            .onAppear {
                audioSamples = Array(repeating: 0, count: 50)
                if coordinator.isRecording {
                    withAnimation {
                        opacity = 1.0
                    }
                }
            }
            .onChange(of: coordinator.transcribedText) { _, _ in
                guard coordinator.isRecording else { return }
                inactivityTimer?.invalidate()
                withAnimation(.spring(response: 0.4, dampingFraction: 0.8, blendDuration: 0.2)) {
                    opacity = 1.0
                }
                inactivityTimer = Timer.scheduledTimer(withTimeInterval: 2.0, repeats: false) { _ in
                    withAnimation(.easeOut(duration: 1.5)) {
                        opacity = 0
                    }
                }
            }
            .onChange(of: coordinator.isRecording) { _, isRecording in
                inactivityTimer?.invalidate()
                if !isRecording {
                    withAnimation(.easeOut(duration: 0.3)) {
                        opacity = 0
                    }
                } else {
                    withAnimation {
                        opacity = 1.0
                    }
                }
            }
            .scaleEffect(x: coordinator.isRecording ? 1.0 : 0.1, anchor: .leading)
            .offset(x: coordinator.isRecording ? 0 : -20)
            .animation(
                caretUIAnimation,
                value: coordinator.isRecording
            )
            .allowsHitTesting(false)
    }
    
    private var waveformContent: some View {
        HStack(spacing: 3) {
            ForEach(0..<audioSamples.count, id: \.self) { index in
                let height = CGFloat(audioSamples[index]) * (coordinator.menuItemSize + 24)
                Rectangle()
                    .frame(width: 2, height: max(0, height))
            }
        }
        .frame(width: 150, height: coordinator.menuItemSize + 16, alignment: .trailing)
        .clipped()
        .foregroundColor(waveformColors)
        .padding(.horizontal, 16)
        .padding(.vertical, 6)
    }
}

struct TranscriptionView: View {
    @ObservedObject var coordinator: CaretUICoordinator
    
    private struct Word: Identifiable, Equatable {
        let id = UUID()
        let text: String
    }
    
    @State private var displayedWords: [Word] = []
    @State private var opacity: Double = 0
    @State private var inactivityTimer: Timer?
    @Environment(\.colorScheme) private var colorScheme
    
    private let caretUIAnimation = Animation.spring(response: 0.3, dampingFraction: 0.8, blendDuration: 0.1)

    private var textColor: Color {
        colorScheme == .dark ? .white : .black
    }

    var body: some View {
        ZStack(alignment: .center) {
            Rectangle()
                .fill(.ultraThinMaterial)
                .blur(radius: 8)
                .frame(
                    width: 250,
                    height: coordinator.menuItemSize
                )
                .allowsHitTesting(false)

            transcriptionContent
        }
        .mask(
            LinearGradient(
                gradient: Gradient(stops: [
                    .init(color: .clear, location: 0.0),
                    .init(color: .black.opacity(0.0), location: 0.1),
                    .init(color: .black.opacity(0.2), location: 0.4),
                    .init(color: .black.opacity(1.0), location: 0.8),
                    .init(color: .black, location: 1.0)
                ]),
                startPoint: .leading,
                endPoint: .trailing
            )
        )
        .scaleEffect(x: coordinator.isRecording ? 1.0 : 0.1, anchor: .trailing)
        .offset(x: coordinator.isRecording ? 0 : 20)
        .opacity(opacity)
        .onAppear {
            if coordinator.isRecording {
                withAnimation {
                    opacity = 1.0
                }
            }
        }
        .onChange(of: coordinator.transcribedText) { oldValue, newValue in
            guard coordinator.isRecording else { return }
            let words = newValue.split(separator: " ").map { String($0) }
            let lastThree = Array(words.suffix(3))
            if lastThree != displayedWords.map({ $0.text }) {
                withAnimation(.spring(response: 0.4, dampingFraction: 0.8, blendDuration: 0.2)) {
                    displayedWords = lastThree.map { Word(text: $0) }
                    opacity = 1.0
                }
                inactivityTimer?.invalidate()
                inactivityTimer = Timer.scheduledTimer(withTimeInterval: 2.0, repeats: false) { _ in
                    withAnimation(.easeOut(duration: 1.5)) {
                        opacity = 0
                    }
                }
            }
        }
        .onChange(of: coordinator.isRecording) { _, isRecording in
            inactivityTimer?.invalidate()
            if !isRecording {
                withAnimation(.easeOut(duration: 0.3)) {
                    opacity = 0
                }
            } else {
                withAnimation {
                    opacity = 1.0
                }
            }
        }
        .animation(
            caretUIAnimation,
            value: coordinator.isRecording
        )
        .allowsHitTesting(false)
    }
    
    private var transcriptionContent: some View {
        Group {
            if displayedWords.isEmpty {
                Text("Listening...".localized)
                    .font(.system(size: coordinator.promptFieldFontSize, weight: .regular))
                    .foregroundColor(textColor)
                    .frame(maxWidth: 270, alignment: .trailing)
            } else {
                HStack(spacing: 5) {
                    ForEach(displayedWords) { word in
                        Text(word.text)
                            .font(.system(size: coordinator.promptFieldFontSize, weight: .regular))
                            .foregroundColor(textColor)
                            .transition(.asymmetric(
                                insertion: .move(edge: .trailing).combined(with: .opacity),
                                removal: .move(edge: .leading).combined(with: .opacity)
                            ))
                    }
                }
            }
        }
        .padding(.horizontal, 16)
        .padding(.vertical, 8)
        .frame(maxWidth: 270, minHeight: coordinator.menuItemSize, alignment: .trailing)
    }
}

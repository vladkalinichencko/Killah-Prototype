import SwiftUI
import AppKit

class SmartCaret: ObservableObject {
    let coordinator: CaretUICoordinator
    
    init(coordinator: CaretUICoordinator) {
        self.coordinator = coordinator
    }
    
    var caretPosition: CGPoint { coordinator.caretPosition }
    var caretSize: CGSize { coordinator.caretSize }
    var isExpanded: Bool { coordinator.isExpanded }
    var editorFontSize: CGFloat { coordinator.editorFontSize }
    
    func update(for textView: NSTextView, at charIndex: Int? = nil, shouldHide: Bool = false) {
        coordinator.updateCaretPosition(for: textView, at: charIndex)
    }
}

struct SmartCaretView: View {
    @ObservedObject var coordinator: CaretUICoordinator

    var body: some View {
        Rectangle()
            .fill(coordinator.caretColor)
            .frame(
                width: coordinator.caretWidth,
                height: coordinator.caretHeight
            )
            .scaleEffect(coordinator.isCaretPressed ? 0.9 : 1.0)
            .shadow(color: coordinator.shadowColor, radius: coordinator.shadowRadius)
            .contentShape(Rectangle().size(width: 20, height: coordinator.caretSize.height + 20))
            .onTapGesture {
                coordinator.toggleExpanded()
            }
            .onHover { hovering in
                withAnimation(.easeInOut(duration: 0.2)) {
                    coordinator.isCaretHovered = hovering
                }
            }
            .pointingHandCursor()
            .onPressGesture(onPress: {
                withAnimation(.easeInOut(duration: 0.2)) {
                    coordinator.isCaretPressed = true
                }
            }, onRelease: {
                withAnimation(.easeInOut(duration: 0.2)) {
                    coordinator.isCaretPressed = false
                }
            })
    }
}

struct CaretRecordButton: View {
    @ObservedObject var coordinator: CaretUICoordinator
    @State private var isPressed = false
    
    var body: some View {
        ZStack {
            RoundedRectangle(cornerRadius: 8)
                .fill(.ultraThinMaterial)
                .overlay(
                    RoundedRectangle(cornerRadius: 8)
                        .stroke(Color.primary.opacity(0.15), lineWidth: 0.75)
                )

            Button(action: {
                coordinator.startRecording()
            }) {
                Circle()
                    .fill(Color.red)
                    .frame(
                        width: coordinator.fixedMenuItemSize * 0.6,
                        height: coordinator.fixedMenuItemSize * 0.6
                    )
                    .scaleEffect(isPressed ? 0.9 : 1.0)
                    .shadow(color: Color.red.opacity(0.4), radius: 4, x: 0, y: 2)
            }
            .buttonStyle(PlainButtonStyle())
            .onPressGesture(
                onPress: { 
                    withAnimation(.easeInOut(duration: 0.1)) {
                        isPressed = true 
                    }
                },
                onRelease: { 
                    withAnimation(.easeInOut(duration: 0.1)) {
                        isPressed = false 
                    }
                }
            )
        }
        .frame(
            width: coordinator.fixedMenuItemSize + 12,
            height: coordinator.fixedMenuItemSize + 12
        )
        .shadow(color: Color.black.opacity(0.1), radius: 10, x: 0, y: 3)
        .scaleEffect(x: coordinator.isExpanded && !coordinator.isRecording ? 1.0 : 0.1, anchor: .trailing)
        .offset(x: coordinator.isExpanded && !coordinator.isRecording ? 0 : coordinator.caretButtonPadding)
        .opacity(coordinator.isExpanded && !coordinator.isRecording ? 1.0 : 0.0)
        .animation(
            .spring(response: 0.3, dampingFraction: 0.8, blendDuration: 0.1), 
            value: coordinator.isExpanded
        )
        .animation(
            .spring(response: 0.3, dampingFraction: 0.8, blendDuration: 0.1), 
            value: coordinator.isRecording
        )
        .pointingHandCursor()
        .allowsHitTesting(coordinator.isExpanded && !coordinator.isRecording)
    }
}

struct CaretPauseButton: View {
    @ObservedObject var coordinator: CaretUICoordinator
    @State private var isPressed = false
    
    var body: some View {
        ZStack {
            RoundedRectangle(cornerRadius: 8)
                .fill(.ultraThinMaterial)
                .overlay(
                    RoundedRectangle(cornerRadius: 8)
                        .stroke(Color.primary.opacity(0.15), lineWidth: 0.75)
                )

            Button(action: {
                coordinator.togglePause()
            }) {
                Image(systemName: coordinator.isPaused ? "play.fill" : "pause.fill")
                    .font(.system(size: coordinator.fixedMenuItemSize * 0.5))
                    .foregroundColor(.primary)
                    .scaleEffect(isPressed ? 0.9 : 1.0)
            }
            .buttonStyle(PlainButtonStyle())
            .onPressGesture(
                onPress: { 
                    withAnimation(.easeInOut(duration: 0.1)) {
                        isPressed = true 
                    }
                },
                onRelease: { 
                    withAnimation(.easeInOut(duration: 0.1)) {
                        isPressed = false 
                    }
                }
            )
        }
        .frame(
            width: coordinator.fixedMenuItemSize + 12,
            height: coordinator.fixedMenuItemSize + 12
        )
        .shadow(color: Color.black.opacity(0.1), radius: 10, x: 0, y: 3)
        .scaleEffect(y: coordinator.isRecording ? 1.0 : 0.1, anchor: .bottom)
        .offset(y: coordinator.isRecording ? 0 : -coordinator.caretButtonPadding)
        .opacity(coordinator.isRecording ? 1.0 : 0.0)
        .animation(
            .interpolatingSpring(stiffness: 300, damping: 20).speed(1.5),
            value: coordinator.isRecording
        )
        .pointingHandCursor()
        .allowsHitTesting(coordinator.isRecording)
    }
}

struct CaretStopButton: View {
    @ObservedObject var coordinator: CaretUICoordinator
    @State private var isPressed = false
    
    var body: some View {
        ZStack {
            RoundedRectangle(cornerRadius: 8)
                .fill(.ultraThinMaterial)
                .overlay(
                    RoundedRectangle(cornerRadius: 8)
                        .stroke(Color.primary.opacity(0.15), lineWidth: 0.75)
                )

            Button(action: {
                coordinator.stopRecording()
            }) {
                Image(systemName: "stop.fill")
                    .font(.system(size: coordinator.fixedMenuItemSize * 0.5))
                    .foregroundColor(.red)
                    .scaleEffect(isPressed ? 0.9 : 1.0)
            }
            .buttonStyle(PlainButtonStyle())
            .onPressGesture(
                onPress: { 
                    withAnimation(.easeInOut(duration: 0.1)) {
                        isPressed = true 
                    }
                },
                onRelease: { 
                    withAnimation(.easeInOut(duration: 0.1)) {
                        isPressed = false 
                    }
                }
            )
        }
        .frame(
            width: coordinator.fixedMenuItemSize + 12,
            height: coordinator.fixedMenuItemSize + 12
        )
        .shadow(color: Color.black.opacity(0.1), radius: 10, x: 0, y: 3)
        .scaleEffect(y: coordinator.isRecording ? 1.0 : 0.1, anchor: .top)
        .offset(y: coordinator.isRecording ? 0 : coordinator.caretButtonPadding)
        .opacity(coordinator.isRecording ? 1.0 : 0.0)
        .animation(
            .interpolatingSpring(stiffness: 300, damping: 20).speed(1.5),
            value: coordinator.isRecording
        )
        .pointingHandCursor()
        .allowsHitTesting(coordinator.isRecording)
    }
}

struct CaretPromptField: View {
    @ObservedObject var coordinator: CaretUICoordinator
    
    var body: some View {
        TextField("Ask lil Pushkin", text: $coordinator.promptText, axis: .vertical)
            .font(.system(size: coordinator.fixedPromptFieldFontSize))
            .textFieldStyle(PlainTextFieldStyle())
            .frame(
                width: coordinator.promptFieldWidth,
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
            .shadow(color: Color.black.opacity(0.1), radius: 10, x: 0, y: 3)
            .scaleEffect(x: coordinator.isExpanded && !coordinator.isRecording ? 1.0 : 0.1, anchor: .leading)
            .offset(
                x: coordinator.isExpanded && !coordinator.isRecording
                    ? coordinator.promptFieldWidthOffset
                    : -coordinator.caretButtonPadding
            )
            .opacity(coordinator.isExpanded && !coordinator.isRecording ? 1.0 : 0.0)
            .onHover { hovering in
                withAnimation(.easeInOut(duration: 0.2)) {
                    coordinator.isPromptFieldHovered = hovering
                }
            }
            .textCursor()
            .animation(
                .spring(response: 0.3, dampingFraction: 0.8, blendDuration: 0.1), 
                value: coordinator.isExpanded
            )
            .animation(
                .spring(response: 0.3, dampingFraction: 0.8, blendDuration: 0.1), 
                value: coordinator.isRecording
            )
            .animation(
                .easeInOut(duration: 0.2),
                value: coordinator.promptText
            )
            .animation(
                .easeInOut(duration: 0.2),
                value: coordinator.promptFieldWidth
            )
            .allowsHitTesting(coordinator.isExpanded && !coordinator.isRecording)
    }
}

struct AudioWaveformView: View {
    @ObservedObject var coordinator: CaretUICoordinator
    @State private var audioSamples: [Float] = Array(repeating: 0, count: 30)

    private let timer = Timer.publish(every: 0.05, on: .main, in: .common).autoconnect()

    var body: some View {
        HStack(spacing: 2) {
            ForEach(0..<audioSamples.count, id: \.self) { index in
                let height = CGFloat(audioSamples[index]) * (coordinator.fixedMenuItemSize + 12)
                RoundedRectangle(cornerRadius: 1.5)
                    .frame(width: 3, height: max(2, height))
            }
        }
        .frame(width: 150, height: coordinator.fixedMenuItemSize + 12, alignment: .trailing)
        .clipped()
        .foregroundStyle(
            LinearGradient(
                gradient: Gradient(colors: [Color.accentColor.opacity(0.7), Color.accentColor]),
                startPoint: .bottom,
                endPoint: .top
            )
        )
        .shadow(color: Color.accentColor.opacity(0.3), radius: 5, y: 2)
        .onReceive(timer) { _ in
            guard coordinator.isRecording && !coordinator.isPaused else { return }
            
            // Shift samples to the left
            audioSamples.removeFirst()
            // Add new sample at the end
            let newSample = coordinator.audioLevel * Float.random(in: 0.7...1.3) // Add some variation
            audioSamples.append(newSample)
        }
        .onAppear {
            // Reset samples when the view appears
            audioSamples = Array(repeating: 0, count: 30)
        }
        .scaleEffect(x: coordinator.isRecording ? 1.0 : 0.1, anchor: .trailing)
        .offset(x: coordinator.isRecording ? 0 : coordinator.caretButtonPadding)
        .opacity(coordinator.isRecording ? 1.0 : 0.0)
        .animation(
            .spring(response: 0.3, dampingFraction: 0.8, blendDuration: 0.1),
            value: coordinator.isRecording
        )
        .allowsHitTesting(false)
    }
}

struct TranscriptionView: View {
    @ObservedObject var coordinator: CaretUICoordinator
    
    var body: some View {
        let textView = Text(coordinator.transcribedText.isEmpty ? "Listening..." : coordinator.transcribedText)
            .font(.system(size: coordinator.fixedPromptFieldFontSize))
            .foregroundColor(.primary)
            .multilineTextAlignment(.trailing)
            .frame(maxWidth: 150, alignment: .trailing)
            .padding(.horizontal, 12)
            .padding(.vertical, 6)
            .shadow(color: .black.opacity(0.3), radius: 2, y: 1) // Text shadow for better readability

        return ZStack {
            // Blur background, making the view itself blurry
            VisualEffectBlur(blurStyle: .popover)
                .clipShape(RoundedRectangle(cornerRadius: 8))
                .overlay(
                    RoundedRectangle(cornerRadius: 8)
                        .stroke(Color.primary.opacity(0.15), lineWidth: 0.75)
                )

            // Fading mask for the text
            textView
                .mask(
                    LinearGradient(
                        gradient: Gradient(stops: [
                            .init(color: .clear, location: 0),
                            .init(color: .black, location: 0.2),
                            .init(color: .black, location: 1.0)
                        ]),
                        startPoint: .leading,
                        endPoint: .trailing
                    )
                )
        }
        .shadow(color: Color.black.opacity(0.15), radius: 10, x: 0, y: 4)
        .scaleEffect(x: coordinator.isRecording ? 1.0 : 0.1, anchor: .trailing)
        .offset(x: coordinator.isRecording ? 0 : coordinator.caretButtonPadding)
        .opacity(coordinator.isRecording ? 1.0 : 0.0)
        .animation(
            .spring(response: 0.3, dampingFraction: 0.8, blendDuration: 0.1),
            value: coordinator.isRecording
        )
        .allowsHitTesting(false)
    }
}

struct VisualEffectBlur: NSViewRepresentable {
    var blurStyle: NSVisualEffectView.Material

    func makeNSView(context: Context) -> NSVisualEffectView {
        let view = NSVisualEffectView()
        view.blendingMode = .behindWindow
        view.state = .active
        view.material = blurStyle
        return view
    }

    func updateNSView(_ nsView: NSVisualEffectView, context: Context) {
        nsView.material = blurStyle
    }
}

extension View {
    func pointingHandCursor() -> some View {
        self.onHover { hovering in
            if hovering {
                NSCursor.pointingHand.push()
            } else {
                NSCursor.pop()
            }
        }
    }
    
    func textCursor() -> some View {
        self.onHover { hovering in
            if hovering {
                NSCursor.iBeam.push()
            } else {
                NSCursor.pop()
            }
        }
    }
    
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

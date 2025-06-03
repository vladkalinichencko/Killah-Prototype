
// filepath: /Users/vladislavkalinichenko/XcodeProjects/Killah Prototype/Killah Prototype/ContentViewVoice.swift
import SwiftUI

struct ContentViewVoice: View {
    @StateObject private var audioManager = AudioInputManager()
    @StateObject private var transcriber = VoiceTranscriberService()
    @StateObject private var llmTransformer = LLMTextTransformer()

    @State private var isRecording = false
    @State private var rawTranscribedText: String = "" // Accumulates for a paragraph
    @State private var transformedTextSegments: [TransformedTextSegment] = []
    @State private var waveformData: [Float] = Array(repeating: 0.0, count: 100) // For visualizer

    var body: some View {
        VStack {
            Text("Voice-to-Text with LLM Transformation")
                .font(.title2)
                .padding()

            WaveformView(waveformData: $waveformData)
                .frame(height: 60)
                .padding(.horizontal)

            Button(action: toggleRecording) {
                Text(isRecording ? "Stop Recording" : "Start Recording")
                    .padding()
                    .frame(maxWidth: .infinity)
                    .background(isRecording ? Color.red : Color.green)
                    .foregroundColor(.white)
                    .cornerRadius(10)
            }
            .padding()

            // Display for raw transcription (could be a temporary pop-up or debug area)
            VStack(alignment: .leading) {
                Text("Live Transcription (Paragraph Buffer):")
                    .font(.caption)
                    .foregroundColor(.gray)
                ScrollView {
                    Text(rawTranscribedText.isEmpty ? "Waiting for speech..." : rawTranscribedText)
                        .frame(maxWidth: .infinity, alignment: .leading)
                        .padding(5)
                }
                .frame(height: 80)
                .background(Color.gray.opacity(0.1))
                .cornerRadius(5)
            }
            .padding(.horizontal)


            Text("Processed & Transformed Text:")
                .font(.headline)
                .padding(.top)
            
            TransformedTextView(segments: $transformedTextSegments)
                .padding(.horizontal)

            Spacer()
        }
        .onAppear(perform: setupCallbacks)
        .onDisappear {
            if isRecording {
                audioManager.stopRecording()
                transcriber.stopService()
                llmTransformer.stopService()
            }
        }
    }

    private func setupCallbacks() {
        audioManager.onAudioChunk = { audioSamples in
            // 1. Update waveform (simplified)
            DispatchQueue.main.async {
                // This needs a more sophisticated way to represent a longer audio signal
                // For now, just show energy of the last chunk
                let rms = audioManager.calculateRMS(audioSamples) // Assuming calculateRMS is public or accessible
                let newWaveData = Array(repeating: rms, count: 20) + self.waveformData.dropLast(20)
                self.waveformData = Array(newWaveData.prefix(100))
            }
            
            // 2. Send to Python for transcription
            transcriber.transcribeAudioChunk(data: audioSamples)
        }

        transcriber.onNewTranscription = { textChunk in
            DispatchQueue.main.async {
                self.rawTranscribedText += textChunk // Append to current paragraph buffer
                
                // Simple heuristic: if the raw text buffer gets long or ends with punctuation,
                // consider it a paragraph and send for transformation.
                // This is where you might have a button later for explicit transformation.
                if self.rawTranscribedText.count > 150 || textChunk.last?.isPunctuation == true {
                    if !self.rawTranscribedText.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
                        llmTransformer.transformParagraph(self.rawTranscribedText.trimmingCharacters(in: .whitespacesAndNewlines))
                        self.rawTranscribedText = "" // Reset for next paragraph
                    }
                }
            }
        }

        llmTransformer.onParagraphTransformed = { original, transformed in
            DispatchQueue.main.async {
                self.transformedTextSegments.append(TransformedTextSegment(original: original, transformed: transformed))
            }
        }
    }

    private func toggleRecording() {
        isRecording.toggle()
        if isRecording {
            transformedTextSegments.removeAll() // Clear previous results
            rawTranscribedText = ""
            audioManager.startRecording()
            // Transcriber and LLM services are assumed to be started/managed by their init or a separate start method if needed
        } else {
            audioManager.stopRecording()
            // Optionally send any remaining rawTranscribedText for transformation
            if !self.rawTranscribedText.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
                 llmTransformer.transformParagraph(self.rawTranscribedText.trimmingCharacters(in: .whitespacesAndNewlines))
                 self.rawTranscribedText = ""
            }
        }
    }
}

// Waveform View (Minimalistic)
struct WaveformView: View {
    @Binding var waveformData: [Float] // Should be normalized amplitude values [-1, 1] or [0, 1]

    var body: some View {
        HStack(spacing: 1) {
            ForEach(0..<waveformData.count, id: \.self) { index in
                Rectangle()
                    .fill(Color.blue.opacity(0.7))
                    .frame(width: 3, height: CGFloat(max(1, abs(waveformData[index] * 50)))) // Scale for display
                    .cornerRadius(1.5)
            }
        }
        .frame(height: 50)
        .background(Color.gray.opacity(0.15))
        .cornerRadius(5)
        .drawingGroup() // Improves performance for frequent updates
    }
}

// Data structure for transformed text segments
struct TransformedTextSegment: Identifiable, Hashable {
    let id = UUID()
    let original: String
    let transformed: String
}

// Transformed Text Display View (Minimalistic Diff)
struct TransformedTextView: View {
    @Binding var segments: [TransformedTextSegment]

    var body: some View {
        ScrollView {
            ScrollViewReader { scrollViewProxy in
                LazyVStack(alignment: .leading, spacing: 12) {
                    ForEach(segments) { segment in
                        VStack(alignment: .leading) {
                            // Simplified diff: Show original crossed out if different, then new one bolded.
                            if segment.original != segment.transformed {
                                Text(segment.original)
                                    .strikethrough(true, color: .red)
                                    .foregroundColor(.gray)
                                    .font(.system(size: 14, design: .monospaced))
                                HStack {
                                    Image(systemName: "arrow.down.right.circle.fill").foregroundColor(.orange)
                                    Text(segment.transformed)
                                        .bold()
                                        .font(.system(size: 15, design: .monospaced))
                                }
                            } else {
                                Text(segment.transformed) // No changes
                                     .font(.system(size: 15, design: .monospaced))
                            }
                        }
                        .padding(10)
                        .frame(maxWidth: .infinity, alignment: .leading)
                        .background(Color.primary.opacity(0.05)) // Subtle background
                        .cornerRadius(8)
                        .id(segment.id)
                    }
                }
                .onChange(of: segments.count) { _ in // Use count to detect new segments
                    if let lastId = segments.last?.id {
                        withAnimation {
                            scrollViewProxy.scrollTo(lastId, anchor: .bottom)
                        }
                    }
                }
            }
        }
        .frame(maxHeight: .infinity)
    }
}

// Minimal Preview
struct ContentViewVoice_Previews: PreviewProvider {
    static var previews: some View {
        ContentViewVoice()
    }
}

// Extension to make AudioInputManager.calculateRMS accessible if it's internal
// If it's already public, this is not needed.
// For the sake of this example, let's assume it might be internal.
extension AudioInputManager {
    func calculateRMS(_ samples: [Float]) -> Float {
        if samples.isEmpty { return 0.0 }
        let sumOfSquares = samples.reduce(0.0) { $0 + ($1 * $1) }
        return sqrt(sumOfSquares / Float(samples.count))
    }
}

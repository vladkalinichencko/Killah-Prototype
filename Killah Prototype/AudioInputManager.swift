import AVFoundation
import Combine

class AudioInputManager: ObservableObject {
    private var audioEngine: AVAudioEngine?
    private var inputNode: AVAudioInputNode?
    // private var audioConverter: AVAudioConverter? // If format conversion is needed

    // For energy-based chunking (simplified)
    private var currentAudioSamples: [Float] = []
    private var silenceFramesCounter: Int = 0
    private let energyThreshold: Float = 0.005 // Example, needs tuning
    private let sampleRate: Double = 16000.0 // Target sample rate for Whisper
    private let bufferSize: UInt32 = 1024 // Example buffer size for tap

    // Heuristic parameters (frames based on bufferSize and sampleRate)
    // e.g. 0.5 seconds of silence = (0.5 * sampleRate) / Double(bufferSize) frames
    private lazy var minSilenceFramesForSplit: Int = Int(0.7 * sampleRate / Double(bufferSize))
    private lazy var minSpeechFramesForChunk: Int = Int(0.1 * sampleRate / Double(bufferSize))
    private lazy var maxSpeechFramesForChunk: Int = Int(5.0 * sampleRate / Double(bufferSize)) // Max 5 sec chunk

    var onAudioChunk: (([Float]) -> Void)? // Sends a chunk of Float samples

    func startRecording() {
        audioEngine = AVAudioEngine()
        inputNode = audioEngine?.inputNode

        guard let inputNode = inputNode else {
            print("Error: Could not get input node.")
            return
        }

        let recordingFormat = AVAudioFormat(commonFormat: .pcmFormatFloat32,
                                            sampleRate: self.sampleRate,
                                            channels: 1,
                                            interleaved: false)

        inputNode.installTap(onBus: 0, bufferSize: bufferSize, format: recordingFormat) { [weak self] (buffer, when) in
            guard let self = self else { return }
            
            let frameLength = Int(buffer.frameLength)
            guard let channelData = buffer.floatChannelData?[0] else { return }
            let samples = Array(UnsafeBufferPointer(start: channelData, count: frameLength))
            
            self.processAudioSamples(samples)
        }

        do {
            audioEngine?.prepare()
            try audioEngine?.start()
            print("AudioInputManager: Recording started.")
        } catch {
            print("Error starting audio engine: \\(error.localizedDescription)")
        }
    }

    private func processAudioSamples(_ samples: [Float]) {
        let rms = calculateRMS(samples)

        if rms > energyThreshold {
            currentAudioSamples.append(contentsOf: samples)
            silenceFramesCounter = 0
        } else { // Below threshold (silence or low noise)
            if !currentAudioSamples.isEmpty { // If there was speech before this silence
                currentAudioSamples.append(contentsOf: samples) // Append some silence too
            }
            silenceFramesCounter += 1
        }

        let currentFrameCount = currentAudioSamples.count / Int(bufferSize) // Approximate frame count based on buffer fills

        // Check for end of speech or max chunk length
        if (silenceFramesCounter >= minSilenceFramesForSplit && currentFrameCount >= minSpeechFramesForChunk) || 
           (currentFrameCount >= maxSpeechFramesForChunk && !currentAudioSamples.isEmpty) {
            
            if !currentAudioSamples.isEmpty {
                print("AudioInputManager: Sending chunk of \\(currentAudioSamples.count) samples.")
                self.onAudioChunk?(currentAudioSamples)
                currentAudioSamples.removeAll()
            }
            silenceFramesCounter = 0 // Reset after sending
        }
    }

    func stopRecording() {
        audioEngine?.stop()
        inputNode?.removeTap(onBus: 0)
        audioEngine = nil
        inputNode = nil
        
        // Send any remaining audio
        if !currentAudioSamples.isEmpty {
            print("AudioInputManager: Sending final chunk of \(currentAudioSamples.count) samples.")
            self.onAudioChunk?(currentAudioSamples)
            currentAudioSamples.removeAll()
        }
        silenceFramesCounter = 0
        print("AudioInputManager: Recording stopped.")
    }

    deinit {
        stopRecording()
    }
}

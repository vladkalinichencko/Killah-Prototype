import Foundation
import AVFoundation
import Combine
import Speech

class AudioEngine: NSObject, ObservableObject, SFSpeechRecognizerDelegate {
    @Published var isRecording = false
    @Published var isPaused = false
    @Published var transcribedText = ""
    @Published var audioLevel: Float = 0.0

    private var audioEngine: AVAudioEngine!
    private var audioFile: AVAudioFile?
    private var audioFilePath: URL?

    // Speech Recognition properties
    private let speechRecognizer = SFSpeechRecognizer(locale: Locale(identifier: "en-US")) // Or your preferred locale
    private var recognitionRequest: SFSpeechAudioBufferRecognitionRequest?
    private var recognitionTask: SFSpeechRecognitionTask?

    override init() {
        super.init()
        audioEngine = AVAudioEngine()
        speechRecognizer?.delegate = self

        SFSpeechRecognizer.requestAuthorization { authStatus in
            OperationQueue.main.addOperation {
                switch authStatus {
                case .authorized:
                    print("Speech recognition authorized")
                case .denied, .restricted, .notDetermined:
                    print("Speech recognition not authorized")
                @unknown default:
                    fatalError()
                }
            }
        }
        
        // Request microphone permission for macOS
        switch AVCaptureDevice.authorizationStatus(for: .audio) {
        case .authorized:
            print("Microphone permission already granted.")
        case .notDetermined:
            AVCaptureDevice.requestAccess(for: .audio) { granted in
                DispatchQueue.main.async {
                    if granted {
                        print("Microphone permission granted.")
                    } else {
                        print("Microphone permission denied.")
                    }
                }
            }
        case .denied, .restricted:
            print("Microphone permission has been denied or restricted.")
        @unknown default:
            // Handle future cases
            break
        }
    }

    func startRecording() {
        guard !isRecording else { return }

        guard AVCaptureDevice.authorizationStatus(for: .audio) == .authorized else {
            print("Microphone permission has not been granted.")
            return
        }

        let inputNode = audioEngine.inputNode
        let recordingFormat = inputNode.outputFormat(forBus: 0)

        recognitionRequest = SFSpeechAudioBufferRecognitionRequest()
        guard let recognitionRequest = recognitionRequest else {
            fatalError("Unable to create an SFSpeechAudioBufferRecognitionRequest object")
        }
        recognitionRequest.shouldReportPartialResults = true
        if #available(macOS 13.0, *) {
            recognitionRequest.requiresOnDeviceRecognition = true
        }

        recognitionTask = speechRecognizer?.recognitionTask(with: recognitionRequest) { [weak self] result, error in
            guard let self = self else { return }
            var isFinal = false

            if let result = result {
                let bestTranscription = result.bestTranscription.formattedString
                DispatchQueue.main.async {
                    self.transcribedText = bestTranscription
                }
                isFinal = result.isFinal
            }

            if error != nil || isFinal {
                self.stopRecording()
            }
        }

        inputNode.installTap(onBus: 0, bufferSize: 1024, format: recordingFormat) { [weak self] (buffer, when) in
            guard let self = self, !self.isPaused else { return }
            self.recognitionRequest?.append(buffer)
            self.updateAudioLevel(buffer: buffer)
        }

        do {
            try audioEngine.start()
            DispatchQueue.main.async {
                self.isRecording = true
                self.isPaused = false
                self.transcribedText = ""
            }
        } catch {
            print("Could not start audio engine: \(error)")
            self.stopRecording()
        }
    }

    func stopRecording() {
        guard isRecording else { return }

        audioEngine.stop()
        audioEngine.inputNode.removeTap(onBus: 0)

        recognitionRequest?.endAudio()
        recognitionTask?.cancel()
        recognitionRequest = nil
        recognitionTask = nil
        
        audioFile = nil

        DispatchQueue.main.async {
            self.isRecording = false
            self.isPaused = false
        }
        audioFilePath = nil
    }

    func togglePause() {
        guard isRecording else { return }
        isPaused.toggle()

        if isPaused {
            audioEngine.pause()
        } else {
            do {
                try audioEngine.start()
            } catch {
                print("Could not restart audio engine after pause: \(error)")
            }
        }
        
        DispatchQueue.main.async {
            self.isPaused = self.isPaused
        }
    }

    private func updateAudioLevel(buffer: AVAudioPCMBuffer) {
        guard let channelData = buffer.floatChannelData else { return }
        let channelDataValue = channelData.pointee
        let channelDataValueArray = UnsafeBufferPointer(start: channelDataValue, count: Int(buffer.frameLength))

        let rms = sqrt(channelDataValueArray.map { $0 * $0 }.reduce(0, +) / Float(buffer.frameLength))
        let avgPower = 20 * log10(rms)
        
        // Normalize to 0-1 range
        let minDb: Float = -60.0 // Adjusted for better sensitivity
        let maxDb: Float = 0.0
        var normalizedLevel = (avgPower - minDb) / (maxDb - minDb)
        normalizedLevel = max(0.0, min(1.0, normalizedLevel))

        DispatchQueue.main.async {
            // Apply a smoothing factor to prevent jerky movements
            self.audioLevel = self.audioLevel * 0.8 + normalizedLevel * 0.2
        }
    }
    
    // This delegate method is called when the availability of the speech recognizer changes
    func speechRecognizer(_ speechRecognizer: SFSpeechRecognizer, availabilityDidChange available: Bool) {
        if available {
            print("Speech recognizer is available")
        } else {
            print("Speech recognizer is not available")
            // Handle unavailability, e.g., disable the record button
        }
    }
}

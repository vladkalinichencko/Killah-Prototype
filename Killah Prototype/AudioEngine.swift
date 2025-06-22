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

    // Timer for periodic transcription UI updates
    private var fullTranscription: String = ""
    private var transcriptionUpdateTimer: Timer?

    // Use AVCaptureSession to trigger macOS microphone indicator
    private var captureSession: AVCaptureSession?
    private var isStopping = false

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
        // Setup AVCaptureSession for microphone indicator
        setupCaptureSession()
    }

    /// Configure AVCaptureSession with audio input to show system mic indicator
    private func setupCaptureSession() {
        let session = AVCaptureSession()
        session.beginConfiguration()
        if let device = AVCaptureDevice.default(for: .audio) {
            do {
                let input = try AVCaptureDeviceInput(device: device)
                if session.canAddInput(input) {
                    session.addInput(input)
                }
            } catch {
                print("Failed to create AVCaptureDeviceInput: \(error)")
            }
        }
        session.commitConfiguration()
        captureSession = session
    }

    func startRecording() {
        guard !isRecording else { return }
        
        // Force microphone permission request
        requestMicrophonePermission { [weak self] granted in
            if granted {
                self?.performRecording()
            } else {
                print("Microphone permission denied")
            }
        }
    }
    
    private func requestMicrophonePermission(completion: @escaping (Bool) -> Void) {
        let status = AVCaptureDevice.authorizationStatus(for: .audio)
        switch status {
        case .authorized:
            completion(true)
        case .notDetermined:
            AVCaptureDevice.requestAccess(for: .audio) { granted in
                DispatchQueue.main.async {
                    completion(granted)
                }
            }
        case .denied, .restricted:
            completion(false)
        @unknown default:
            completion(false)
        }
    }
    
    private func performRecording() {
        // Start AVCaptureSession to trigger system mic indicator
        captureSession?.startRunning()
        
        let inputNode = audioEngine.inputNode
        let recordingFormat = inputNode.outputFormat(forBus: 0)
        
        recognitionRequest = SFSpeechAudioBufferRecognitionRequest()
        guard let recognitionRequest = recognitionRequest else {
            print("Failed to create recognition request")
            return
        }
        recognitionRequest.shouldReportPartialResults = true
        if #available(macOS 13.0, *) {
            recognitionRequest.requiresOnDeviceRecognition = true
        }

        recognitionTask = speechRecognizer?.recognitionTask(with: recognitionRequest) { [weak self] result, error in
            guard let self = self else { return }
            
            if let error = error {
                print("Recognition error: \(error.localizedDescription)")
                DispatchQueue.main.async {
                    self.stopRecording()
                }
                return
            }
            
            var isFinal = false

            if let result = result {
                // Update the full transcription string in the background
                self.fullTranscription = result.bestTranscription.formattedString
                isFinal = result.isFinal
            }

            if isFinal {
                self.updateDisplayedTranscription() // Perform one final update
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
                self.fullTranscription = "" // Reset on start

                // Invalidate any existing timer and start a new one
                self.transcriptionUpdateTimer?.invalidate()
                self.transcriptionUpdateTimer = Timer.scheduledTimer(withTimeInterval: 0.4, repeats: true) { [weak self] _ in
                    self?.updateDisplayedTranscription()
                }
            }
        } catch {
            print("Could not start audio engine: \(error.localizedDescription)")
            if let nsError = error as NSError? {
                print("Error domain: \(nsError.domain), code: \(nsError.code)")
                print("User info: \(nsError.userInfo)")
            }
            self.stopRecording()
        }
    }

    func stopRecording() {
        guard isRecording, !isStopping else { return }

        isStopping = true

        transcriptionUpdateTimer?.invalidate()
        transcriptionUpdateTimer = nil

        // Stop AVCaptureSession to hide system mic indicator
        captureSession?.stopRunning()
        
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
            self.isStopping = false
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
        guard let channelData = buffer.floatChannelData else { 
            print("No channel data available")
            return 
        }
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
    
    private func updateDisplayedTranscription() {
        let words = fullTranscription.split(separator: " ")
        let lastThreeWords = words.suffix(3).joined(separator: " ")
        
        DispatchQueue.main.async {
            if self.transcribedText != lastThreeWords {
                self.transcribedText = lastThreeWords
            }
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

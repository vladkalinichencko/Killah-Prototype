import Foundation
import SwiftUICore
import AVFoundation
import Combine
import Speech

class AudioEngine: NSObject, ObservableObject, SFSpeechRecognizerDelegate {
    @Published var isRecording = false
    @Published var isPaused = false
    @Published var transcribedText = ""
    @Published var audioLevel: Float = 0.0
    
    
    private var savedAudioFilePath: URL?
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
    
    private let llmEngine: LLMEngine // Inject LLMEngine
    private var stateCancellable: AnyCancellable?


    init(llmEngine: LLMEngine) {
        self.llmEngine = llmEngine
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
        
        // Observe LLMEngine state
        stateCancellable = llmEngine.$engineState.sink {state in
            if case .error(let message) = state {
                DispatchQueue.main.async {
                    print("AI Engine Error: \(message)")
                }
            }
        }
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
        print("Starting recording process...")
        
        //llmEngine.abortSuggestion(for: "audio", notifyPython: true)

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
        print("Attempting to access microphone...")
        // Start AVCaptureSession to trigger system mic indicator
        captureSession?.startRunning()
        
        let inputNode = audioEngine.inputNode
        let recordingFormat = inputNode.outputFormat(forBus: 0)
        
        print("Audio format: \(recordingFormat)")
        
        // Path to save the format
        let documentsPath = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask).first!
        print("Documents directory: \(documentsPath.absoluteString)")
        if FileManager.default.isWritableFile(atPath: documentsPath.path) {
            print("✅ Documents directory is writable")
        } else {
            print("❌ Documents directory is NOT writable")
        }
        
        let audioFileName = "recording_\(Date().timeIntervalSince1970).wav"
        audioFilePath = documentsPath.appendingPathComponent(audioFileName)
        print("Audio file path set to: \(audioFilePath!.absoluteString)")
        savedAudioFilePath = audioFilePath // Сохраняем копию
        print("Audio file path set to: \(audioFilePath!.absoluteString)")
        // Creation AVAudioFile for recording
        do {
            audioFile = try AVAudioFile(forWriting: audioFilePath!, settings: recordingFormat.settings)
            print("✅ AVAudioFile created successfully at: \(audioFilePath!.absoluteString)")
        } catch {
            print("❌ Failed to create audio file: \(error)")
            audioFilePath = nil // Явно сбрасываем путь при ошибке
            return
        }
        
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
                print("Transcription completed")
                self.updateDisplayedTranscription() // Perform one final update
                self.stopRecording()
            }
        }

        print("Installing tap on input node...")
        inputNode.installTap(onBus: 0, bufferSize: 1024, format: recordingFormat) { [weak self] (buffer, when) in
            guard let self = self, !self.isPaused else { return }
            
            // Writing buffer to the file
            do {
                try self.audioFile?.write(from: buffer)
            } catch {
                print("Failed to write audio buffer to file: \(error)")
            }
            
            self.recognitionRequest?.append(buffer)
            self.updateAudioLevel(buffer: buffer)
        }

        print("Starting audio engine...")
        do {
            try audioEngine.start()
            print("Audio engine started successfully")
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
                print("Recording state updated: isRecording = true")
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
        guard isRecording else {
            print("⚠️ stopRecording called but isRecording is false")
            return
        }

        print("Stopping recording...")

        transcriptionUpdateTimer?.invalidate()
        transcriptionUpdateTimer = nil

        // Stop AVCaptureSession to hide system mic indicator
        captureSession?.stopRunning()
        
        audioEngine.stop()
        audioEngine.inputNode.removeTap(onBus: 0)

        recognitionRequest?.endAudio()
        DispatchQueue.main.asyncAfter(deadline: .now() + 1.0) {
            self.recognitionTask?.cancel()
            self.recognitionRequest = nil
            self.recognitionTask = nil
        }
        
        audioFile = nil

        DispatchQueue.main.async {
            self.isRecording = false
            self.isPaused = false
            if let path = self.savedAudioFilePath ?? self.audioFilePath {
                print("Audio file path: \(path.absoluteString)")
                if FileManager.default.fileExists(atPath: path.path) {
                    print("✅ Audio file exists at: \(path.absoluteString)")
                    do {
                        let attributes = try FileManager.default.attributesOfItem(atPath: path.path)
                        let fileSize = attributes[.size] as? Int64 ?? 0
                        print("File size: \(fileSize) bytes")
                        if fileSize > 44 { // WAV заголовок ~44 байта
                            self.audioFilePath = path // Восстанавливаем путь
                            self.processAudioFileWithPython()
                        } else {
                            print("⚠️ Audio file is too small (likely empty): \(fileSize) bytes")
                        }
                    } catch {
                        print("❌ Error checking file attributes: \(error)")
                    }
                } else {
                    print("❌ Audio file does NOT exist at: \(path.absoluteString)")
                }
            } else {
                print("❌ Both savedAudioFilePath and audioFilePath are nil")
            }
            self.audioFilePath = nil
            self.savedAudioFilePath = nil
        }
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

        // Debug: Print audio level occasionally to verify we're getting data
        if Int.random(in: 1...100) == 1 { // Print 1% of the time to avoid spam
            print("Audio level: \(normalizedLevel), RMS: \(rms), Power: \(avgPower)")
        }

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
    
    private func processAudioFileWithPython() {
        guard let audioFilePath = audioFilePath else {
            print("No audio file available to process")
            return
        }
        
        let checkInterval: TimeInterval = 0.1
        var attempts = 50
        print("Checking audio.py state for processing %s", audioFilePath.path)
        
        func trySendData() {
            let runnerState = llmEngine.getRunnerState(for: "audio")
            print("Current audio.py state: %s", String(describing: runnerState))
            if runnerState == .running {
                llmEngine.generateSuggestion(
                    for: "audio",
                    prompt: audioFilePath.path,
                    tokenStreamCallback: { token in
                        print("Audio token received: %s", token)
                    },
                    onComplete: { [weak self] result in
                        guard self != nil else { return }
                        switch result {
                        case .success(let output):
                            print("Audio processing completed: %s", output)
                            // Удаляем аудиофайл после успешной обработки
                            do {
                                try FileManager.default.removeItem(at: audioFilePath)
                                print("✅ Audio file deleted: \(audioFilePath.path)")
                            } catch {
                                print("❌ Failed to delete audio file \(audioFilePath.path): \(error)")
                            }
                        case .failure(let error):
                            print("Audio processing failed: %s", error.localizedDescription)
                            DispatchQueue.main.async {
                                print("Failed to process audio: %s", error.localizedDescription)
                            }
                        }
                    }
                )
            } else if attempts > 0 {
                attempts -= 1
                DispatchQueue.main.asyncAfter(deadline: .now() + checkInterval) {
                    trySendData()
                }
            } else {
                print("❌ audio.py failed to reach running state after %.1f seconds", 50 * checkInterval)
                DispatchQueue.main.async {
                    print("Audio processing engine not ready")
                }
            }
        }
        
        trySendData()
    }
        
}

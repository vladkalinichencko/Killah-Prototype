import Foundation
import Combine

class VoiceTranscriberService: ObservableObject {
    private var process: Process?
    private var stdinPipe: Pipe?
    private var stdoutPipe: Pipe?

    var onNewTranscription: ((String) -> Void)?

    // Queue for managing audio chunks to be transcribed
    private let transcriptionQueue = DispatchQueue(label: "com.example.transcriptionQueue", qos: .userInitiated)
    private var isPythonProcessBusy = false // Flag to indicate if Python is currently processing

    init() {
        setupPythonProcess()
    }

    private func setupPythonProcess() {
        // This is a placeholder. In a real app, you'd manage the Python script lifecycle.
        // For now, we assume it's always running or started on demand.
        // The actual process launching and communication would be similar to LLMEngine.
        // When Python script is ready, set isPythonProcessBusy = false
        print("VoiceTranscriberService: Python process setup (simulated).")
        // For simulation, assume it's ready immediately
        self.isPythonProcessBusy = false
    }

    func transcribeAudioChunk(data: [Float]) {
        print("VoiceTranscriberService: Received audio chunk of size \\(data.count). Queuing for transcription.")
        
        transcriptionQueue.async { [weak self] in
            guard let self = self else { return }
            
            // This block ensures that we wait if Python is busy
            // and process chunks one by one from the queue.
            
            // Wait until Python process is not busy.
            // A more robust solution might use a DispatchSemaphore or OperationQueue dependencies.
            // For simplicity, this basic check inside a serial queue works if calls to
            // _processChunkOnQueue are themselves serialized.
            // However, a simple flag isn't enough if multiple items get queued rapidly.
            // The serial queue itself ensures one-by-one execution of these async blocks.
            
            // The critical part is that _actuallySendToPythonAndProcessResponse
            // should only be called when the previous call has completed.
            
            self._processChunkOnQueue(audioData: data)
        }
    }

    private func _processChunkOnQueue(audioData: [Float]) {
        // This method is always called on `transcriptionQueue` serially.
        
        // Simple busy check (could be more sophisticated with semaphores for true backpressure)
        if self.isPythonProcessBusy {
            print("VoiceTranscriberService: Python busy, re-queuing chunk (or rather, this task will wait its turn on the serial queue).")
            // With a serial queue, this task just waits. If it were concurrent,
            // we'd need to explicitly re-queue or use a semaphore.
            // For now, let's assume the serial queue handles the "waiting".
            // To make it more explicit that we wait for Python to be free:
            // We'd need a loop with a condition variable or semaphore here if we weren't on a serial queue.
            // Since we are, the next task on the queue won't start until this one finishes.
        }

        self.isPythonProcessBusy = true
        print("VoiceTranscriberService: Processing chunk of size \\(audioData.count) from queue...")

        // --- Actual Python Interaction (Simulated) ---
        // 1. Convert [Float] to Data (if Python script expects bytes)
        // 2. Send data to Python's stdin.
        // 3. Read transcribed text from Python's stdout.
        // This part needs to be synchronous within this block or use a completion handler
        // to signal when Python is free again.

        // Simulate sending data to Python and waiting for a response
        DispatchQueue.global().asyncAfter(deadline: .now() + 2.0) { [weak self] in // Simulate 2s transcription time
            guard let self = self else { return }
            
            let simulatedText = "Transcribed: [chunk_size:\\(audioData.count)] - \\(Date().timeIntervalSince1970). "
            
            DispatchQueue.main.async { // Call UI updates on main thread
                self.onNewTranscription?(simulatedText)
            }
            
            // Signal that Python is now free
            self.transcriptionQueue.async { // Ensure this flag is updated on the queue
                self.isPythonProcessBusy = false
                print("VoiceTranscriberService: Python process now free.")
                // If there are more items in `transcriptionQueue`, the next one will start.
            }
        }
        // --- End Actual Python Interaction (Simulated) ---
    }

    func stopService() {
        transcriptionQueue.sync {
            // Cancel any pending work if possible (tricky with just async blocks)
            // For a real app, OperationQueue would be better for cancellation.
            self.isPythonProcessBusy = true // Prevent new tasks from starting
        }
        // Terminate Python process, close pipes
        process?.terminate()
        process = nil
        stdinPipe = nil
        stdoutPipe = nil
        print("VoiceTranscriberService: Stopped (simulated).")
    }
    
    deinit {
        stopService()
    }
}

import Foundation
import Combine
import AppKit

class LLMEngine: ObservableObject {
    @Published var suggestion: String = ""
    @Published var engineState: EngineState = .idle

    var task: Process?
    private var stdinPipe: Pipe?
    private var stdoutPipe: Pipe?
    private var stderrPipe: Pipe?
    private var cancellables = Set<AnyCancellable>()
    private var lastSentPrompt: String?
    private var outputBuffer = ""

    private var currentTokenCallback: ((String) -> Void)?
    private var currentCompletionCallback: ((Result<String, LLMError>) -> Void)?
    private var accumulatedSuggestion: String = ""
    private var isAbortedManually: Bool = false

    enum EngineState: Equatable {
        case idle
        case starting
        case running
        case stopped
        case error(String)
        static func == (lhs: EngineState, rhs: EngineState) -> Bool {
            switch (lhs, rhs) {
            case (.idle, .idle): return true
            case (.starting, .starting): return true
            case (.running, .running): return true
            case (.stopped, .stopped): return true
            case let (.error(lhsError), .error(rhsError)): return lhsError == rhsError
            default: return false
            }
        }
    }

    init() {
        print("LLMEngine init")
        NotificationCenter.default.publisher(for: NSApplication.willTerminateNotification)
            .sink { [weak self] _ in
                print("App is terminating, stopping engine...")
                self?.stopEngine()
            }
            .store(in: &cancellables)
    }

    private func updateEngineState(_ newState: EngineState) {
        DispatchQueue.main.async {
            if self.engineState != newState {
                print("⚙️ LLMEngine state changing from \(self.engineState) to \(newState)")
                self.engineState = newState
            }
        }
    }

    func startEngine() {
        guard engineState == .idle || engineState == .stopped || (engineState == .error("") && isErrorState(engineState)) else {
            print("ℹ️ Engine is not in a state to be started (current: \(engineState))")
            return
        }

        print("🚀 Attempting to start Python engine...")
        updateEngineState(.starting)
        suggestion = ""
        isAbortedManually = false
        accumulatedSuggestion = ""

        let process = Process()
        task = process

        guard let resourcesPath = Bundle.main.resourcePath else {
            updateEngineState(.error("Bundle resources path not found"))
            return
        }

        let venvPythonPath = resourcesPath + "/venv/bin/python3"
        guard FileManager.default.fileExists(atPath: venvPythonPath) else {
            print("❌ Python binary not found at: \(venvPythonPath)")
            updateEngineState(.error("Python binary not found"))
            return
        }
        print("🐍 Python interpreter path: \(venvPythonPath)")

        let scriptPath = resourcesPath + "/autocomplete.py"
        guard FileManager.default.fileExists(atPath: scriptPath) else {
            print("❌ Python script not found at: \(scriptPath)")
            updateEngineState(.error("autocomplete.py not found"))
            return
        }
        print("📜 Python script path: \(scriptPath)")

        let modelPath = resourcesPath + "/minillm_export.pt"
        if FileManager.default.fileExists(atPath: modelPath) {
            print("📊 Model file found at: \(modelPath)")
        } else {
            print("⚠️ Model file not found at: \(modelPath)")
        }

        process.executableURL = URL(fileURLWithPath: venvPythonPath)
        process.arguments = [scriptPath]

        stdinPipe = Pipe()
        stdoutPipe = Pipe()
        stderrPipe = Pipe()

        process.standardInput = stdinPipe
        process.standardOutput = stdoutPipe
        process.standardError = stderrPipe

        setupStdErrHandler()
        setupStdOutHandler()
        setupTerminationHandler()

        do {
            print("▶️ Launching Python process: \(venvPythonPath) \(scriptPath)")
            try process.run()
            print("✅ Python process launched. PID: \(process.processIdentifier). Waiting for ready signal...")
        } catch {
            print("🫩 Error launching Python process: \(error)")
            updateEngineState(.error("Launch fail: \(error.localizedDescription)"))
            task = nil
            currentTokenCallback = nil
            currentCompletionCallback?(.failure(.processLaunchError(error.localizedDescription)))
            currentCompletionCallback = nil
        }
    }

    private func isErrorState(_ state: EngineState) -> Bool {
        if case .error(_) = state { return true }
        return false
    }

    func generateSuggestion(prompt: String, tokenStreamCallback: @escaping (String) -> Void, onComplete: @escaping (Result<String, LLMError>) -> Void) {
        print("📄 generateSuggestion called with prompt: \"\(prompt)\"")
        guard let runningTask = task, runningTask.isRunning, let stdin = stdinPipe else {
            print("❌ Engine not running or stdin not available. Cannot send prompt. Current state: \(engineState)")
            onComplete(.failure(.engineNotRunning))
            return
        }

        guard engineState == .running else {
            print("⏳ Python script not fully ready yet (State: \(engineState)). Prompt '\(prompt)' not sent.")
            onComplete(.failure(.pythonScriptNotReady))
            return
        }

        if currentTokenCallback != nil || currentCompletionCallback != nil {
            print("ℹ️ Aborting previous suggestion to start new one for prompt: \"\(prompt)\"")
            abortCurrentSuggestion(notifyPython: false)
        }

        lastSentPrompt = prompt
        currentTokenCallback = tokenStreamCallback
        currentCompletionCallback = onComplete
        accumulatedSuggestion = ""
        isAbortedManually = false

        print("➡️ Sending prompt to Python: \"\(prompt)\"")
        guard let data = (prompt + "\n").data(using: .utf8) else {
            print("❌ Error encoding prompt to UTF-8 data.")
            currentCompletionCallback?(.failure(.promptEncodingError))
            currentTokenCallback = nil
            currentCompletionCallback = nil
            return
        }

        let stdinHandle = stdin.fileHandleForWriting
        do {
            if #available(macOS 10.15.4, *) {
                try stdinHandle.write(contentsOf: data)
            } else {
                stdinHandle.write(data)
            }
        } catch {
            print("🫩 Error writing to Python stdin: \(error)")
            engineState = .error("Error writing to Python: \(error.localizedDescription)")
            currentCompletionCallback?(.failure(.stdinWriteError(error.localizedDescription)))
            currentTokenCallback = nil
            currentCompletionCallback = nil
        }
    }

    func abortCurrentSuggestion(notifyPython: Bool = true) {
        print("ℹ️ Aborting current suggestion stream. Notify Python: \(notifyPython). Current state: \(engineState)")
        isAbortedManually = true

        if notifyPython, let runningTask = task, runningTask.isRunning, let stdin = stdinPipe {
            print("➡️ Sending abort signal (empty line) to Python stdin.")
            guard let data = "\n".data(using: .utf8) else {
                print("❌ Error encoding abort signal (empty line) to UTF-8 data.")
                currentCompletionCallback?(.failure(.promptEncodingError))
                return
            }
            let stdinHandle = stdin.fileHandleForWriting
            do {
                if #available(macOS 10.15.4, *) {
                    try stdinHandle.write(contentsOf: data)
                } else {
                    stdinHandle.write(data)
                }
            } catch {
                print("🫩 Error writing abort signal to Python stdin: \(error)")
                currentCompletionCallback?(.failure(.stdinWriteError(error.localizedDescription)))
            }
        }

        if let callback = currentCompletionCallback {
            callback(.failure(.aborted))
        }

        currentTokenCallback = nil
        currentCompletionCallback = nil
        accumulatedSuggestion = ""
    }

    func stopEngine() {
        print("🛑 Attempting to stop Python engine. Current state: \(engineState)")

        if currentTokenCallback != nil || currentCompletionCallback != nil {
            print("ℹ️ Active suggestion generation during stopEngine. Aborting it.")
            abortCurrentSuggestion(notifyPython: true)
        }

        cancellables.forEach { $0.cancel() }
        cancellables.removeAll()

        if let stdin = stdinPipe {
            do {
                try stdin.fileHandleForWriting.close()
                print("🚪 stdin pipe closed.")
            } catch {
                print("⚠️ Error closing stdin pipe: \(error)")
            }
        }

        if let runningTask = task, runningTask.isRunning {
            print("⏳ Terminating Python process PID \(runningTask.processIdentifier)...")
            runningTask.terminate()
        }
        task = nil

        stdoutPipe?.fileHandleForReading.readabilityHandler = nil
        stderrPipe?.fileHandleForReading.readabilityHandler = nil
        stdinPipe = nil
        stdoutPipe = nil
        stderrPipe = nil

        updateEngineState(.stopped)
        DispatchQueue.main.async { self.suggestion = "" }
        isAbortedManually = false
        accumulatedSuggestion = ""
        print("Engine stopped successfully.")
    }

    private func setupStdErrHandler() {
        guard let errPipe = stderrPipe else {
            print("❌ Cannot setup stderr handler: pipe is nil.")
            return
        }
        let errHandle = errPipe.fileHandleForReading
        errHandle.readabilityHandler = { [weak self] pipe in
            guard let self = self else { return }
            let data = pipe.availableData
            if data.isEmpty {
                print("🐍 STDERR: EOF reached or pipe closed.")
                if !(self.task?.isRunning ?? false) { return }
            } else {
                let rawOutput = String(data: data, encoding: .utf8) ?? "<failed to decode stderr as utf8>"
                DispatchQueue.main.async {
                    print("🐍 STDERR RAW: \"\(rawOutput.trimmingCharacters(in: .whitespacesAndNewlines))\"")
                    let lines = rawOutput.components(separatedBy: .newlines).filter { !$0.isEmpty }
                    for line in lines {
                        print("🐍 STDERR Line: \"\(line)\"")
                    }
                }
            }
            if self.task?.isRunning ?? false {
                pipe.waitForDataInBackgroundAndNotify()
            }
        }
        print("🔧 STDERR handler setup.")
    }

    private func setupStdOutHandler() {
        guard let outPipe = stdoutPipe else {
            print("❌ Cannot setup stdout handler: pipe is nil.")
            return
        }
        let outHandle = outPipe.fileHandleForReading

        outHandle.readabilityHandler = { [weak self] pipe in
            guard let self = self else { return }
            let data = pipe.availableData
            if !data.isEmpty {
                if let rawOutput = String(data: data, encoding: .utf8) {
                    outputBuffer += rawOutput
                    var lines = outputBuffer.components(separatedBy: .newlines)
                    outputBuffer = lines.removeLast() // Сохраняем неполную строку

                    for line in lines where !line.isEmpty {
                        DispatchQueue.main.async {
                            if self.engineState == .starting && line == "Entering main processing loop." {
                                print("✅ Python script is ready and processing loop entered.")
                                self.updateEngineState(.running)
                                self.isAbortedManually = false
                            } else if self.engineState == .running {
                                if line == "Streaming suggestions..." {
                                    self.accumulatedSuggestion = ""
                                } else if line == "END_SUGGESTIONS" {
                                    if let callback = self.currentCompletionCallback {
                                        callback(.success(self.accumulatedSuggestion))
                                    }
                                    self.currentTokenCallback = nil
                                    self.currentCompletionCallback = nil
                                    self.accumulatedSuggestion = ""
                                    self.isAbortedManually = false
                                } else if !self.isAbortedManually && line != "Entering main processing loop." {
                                    if let callback = self.currentTokenCallback {
                                        callback(line)
                                        self.accumulatedSuggestion += line
                                    }
                                }
                            }
                        }
                    }
                }
            }
            if self.task?.isRunning ?? false {
                pipe.waitForDataInBackgroundAndNotify()
            }
        }
        outHandle.waitForDataInBackgroundAndNotify()
        print("🔧 STDOUT handler setup and listening using readabilityHandler.")
    }

    private func setupTerminationHandler() {
        task?.terminationHandler = { [weak self] process in
            guard let self = self else { return }
            DispatchQueue.main.async {
                print("🔄 Python process terminated. PID: \(process.processIdentifier), Exit Code: \(process.terminationStatus), Reason: \(process.terminationReason.rawValue). Current engine state: \(self.engineState)")
                let previousState = self.engineState
                if previousState != .stopped && !self.isAbortedManually {
                    let errorMsg = "Python process terminated unexpectedly (Exit code: \(process.terminationStatus))."
                    print("‼️ \(errorMsg)")
                    self.updateEngineState(.error(errorMsg))
                    if let callback = self.currentCompletionCallback {
                        callback(.failure(.scriptError(errorMsg)))
                    }
                } else if previousState == .running || previousState == .starting {
                    print("ℹ️ Python process terminated, likely as part of normal stop or script completion.")
                    self.updateEngineState(.stopped)
                }
                self.currentTokenCallback = nil
                self.currentCompletionCallback = nil
                self.accumulatedSuggestion = ""
                if self.task != nil { self.task = nil }
            }
        }
        print("🔧 Process termination handler setup.")
    }

    deinit {
        print("🗑️ LLMEngine deinit - Stopping engine.")
        stopEngine()
    }
}

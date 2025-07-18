import Foundation
import Combine

protocol PythonScriptRunner {
    var scriptName: String { get }
    var state: LLMEngine.EngineState { get }
    func start()
    func sendData(_ data: String, tokenStreamCallback: @escaping (String) -> Void, onComplete: @escaping (Result<String, LLMEngine.LLMError>) -> Void)
    func sendCommand(_ command: String)
    func stop()
    func abortSuggestion(notifyPython: Bool)
}

class BaseScriptRunner: NSObject, PythonScriptRunner {
    let scriptName: String
    private let modelDirectory: String?
    private var _state: LLMEngine.EngineState = .idle
    private var task: Process?
    private var stdinPipe: Pipe?
    private var stdoutPipe: Pipe?
    private var stderrPipe: Pipe?
    private var cancellables = Set<AnyCancellable>()
    private var outputBuffer = ""
    private var currentTokenCallback: ((String) -> Void)?
    private var currentCompletionCallback: ((Result<String, LLMEngine.LLMError>) -> Void)?
    private var accumulatedOutput: String = ""
    private var isAbortedManually: Bool = false

    init(scriptName: String, modelDirectory: String?) {
        self.scriptName = scriptName
        self.modelDirectory = modelDirectory
        super.init()
    }

    var state: LLMEngine.EngineState {
        return _state
    }

    func start() {
        guard state == .idle || state == .stopped || isErrorState(state) else {
            print("ℹ️ \(scriptName) is not in a state to be started (current: \(state))")
            return
        }

        print("🚀 Starting \(scriptName)...")
        updateState(.starting)

        let process = Process()
        task = process

        guard let resourcesPath = Bundle.main.resourcePath else {
            updateState(.error("Bundle resources path not found"))
            return
        }

        let venvPythonPath = resourcesPath + "/venv/bin/python3"
        guard FileManager.default.fileExists(atPath: venvPythonPath) else {
            print("❌ Python binary not found at: \(venvPythonPath)")
            updateState(.error("Python binary not found"))
            return
        }

        let scriptPath = resourcesPath + "/\(scriptName)"
        guard FileManager.default.fileExists(atPath: scriptPath) else {
            print("❌ Script not found at: \(scriptPath)")
            updateState(.error("\(scriptName) not found"))
            return
        }

        process.executableURL = URL(fileURLWithPath: venvPythonPath)
        
        var env = ProcessInfo.processInfo.environment
        if let modelDir = modelDirectory {
            env["MODEL_DIR"] = modelDir
        }
        
        // Load HF_TOKEN from config.env and pass to Python scripts
        if let hfToken = loadHFToken() {
            env["HF_TOKEN"] = hfToken
        }
        
        // Force single-threaded execution for compatibility with macOS sandboxing
        // and to prevent joblib from trying to spawn processes.
        env["OMP_NUM_THREADS"] = "1"
        env["TOKENIZERS_PARALLELISM"] = "false"
        
        process.environment = env
        
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
            print("▶️ Launching \(scriptName): \(venvPythonPath) \(scriptPath)")
            try process.run()
            print("✅ \(scriptName) launched. PID: \(process.processIdentifier)")
        } catch {
            print("🫩 Error launching \(scriptName): \(error)")
            updateState(.error("Launch fail: \(error.localizedDescription)"))
            task = nil
            currentTokenCallback = nil
            currentCompletionCallback?(.failure(.processLaunchError(error.localizedDescription)))
            currentCompletionCallback = nil
        }
    }

    func sendData(_ data: String, tokenStreamCallback: @escaping (String) -> Void, onComplete: @escaping (Result<String, LLMEngine.LLMError>) -> Void) {
        guard let runningTask = task, runningTask.isRunning, let stdin = stdinPipe else {
            print("❌ \(scriptName) not running or stdin not available. Current state: \(state)")
            onComplete(.failure(.engineNotRunning))
            return
        }

        guard state == .running else {
            print("⏳ \(scriptName) not ready yet (State: \(state)). Data not sent.")
            onComplete(.failure(.pythonScriptNotReady))
            return
        }

        if currentTokenCallback != nil || currentCompletionCallback != nil {
            print("ℹ️ Aborting previous suggestion for \(scriptName) before sending new data")
            abortSuggestion(notifyPython: true)
        }

        currentTokenCallback = tokenStreamCallback
        currentCompletionCallback = onComplete
        accumulatedOutput = ""
        isAbortedManually = false

        print("➡️ Sending data to \(scriptName): \"\(data.suffix(100))\"")
        guard let inputData = (data + "\n").data(using: .utf8) else {
            print("❌ Error encoding data to UTF-8.")
            currentCompletionCallback?(.failure(.promptEncodingError))
            currentTokenCallback = nil
            currentCompletionCallback = nil
            return
        }

        let stdinHandle = stdin.fileHandleForWriting
        do {
            if #available(macOS 10.15.4, *) {
                try stdinHandle.write(contentsOf: inputData)
            } else {
                stdinHandle.write(inputData)
            }
        } catch {
            print("🫩 Error writing to \(scriptName) stdin: \(error)")
            updateState(.error("Error writing to Python: \(error.localizedDescription)"))
            currentCompletionCallback?(.failure(.stdinWriteError(error.localizedDescription)))
            currentTokenCallback = nil
            currentCompletionCallback = nil
        }
    }

    func abortSuggestion(notifyPython: Bool) {
        print("ℹ️ Aborting suggestion for \(scriptName). Notify Python: \(notifyPython). Current state: \(state)")
        isAbortedManually = true

        if notifyPython, let runningTask = task, runningTask.isRunning, let stdin = stdinPipe {
            print("➡️ Sending abort signal to \(scriptName) stdin")
            guard let data = "\n".data(using: .utf8) else {
                print("❌ Error encoding abort signal")
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
                print("🫩 Error writing abort signal to \(scriptName) stdin: \(error)")
                currentCompletionCallback?(.failure(.stdinWriteError(error.localizedDescription)))
            }
        }

        if let callback = currentCompletionCallback {
            callback(.failure(.aborted))
        }

        currentTokenCallback = nil
        currentCompletionCallback = nil
        accumulatedOutput = ""
    }

    func stop() {
        print("🛑 Stopping \(scriptName). Current state: \(state)")
        print(Thread.callStackSymbols.joined(separator: "\n"))

        if currentTokenCallback != nil || currentCompletionCallback != nil {
            print("ℹ️ Active task in \(scriptName). Aborting it.")
            abortSuggestion(notifyPython: true)
        }

        cancellables.forEach { $0.cancel() }
        cancellables.removeAll()

        if let stdin = stdinPipe {
            do {
                try stdin.fileHandleForWriting.close()
                print("🚪 \(scriptName) stdin pipe closed.")
            } catch {
                print("⚠️ Error closing \(scriptName) stdin pipe: \(error)")
            }
        }

        if let runningTask = task, runningTask.isRunning {
            print("⏳ Terminating \(scriptName) process PID \(runningTask.processIdentifier)...")
            runningTask.terminate()
        }
        task = nil

        stdoutPipe?.fileHandleForReading.readabilityHandler = nil
        stderrPipe?.fileHandleForReading.readabilityHandler = nil
        stdinPipe = nil
        stdoutPipe = nil
        stderrPipe = nil

        updateState(.stopped)
        accumulatedOutput = ""
        print("\(scriptName) stopped successfully.")
    }

    private func updateState(_ newState: LLMEngine.EngineState) {
        DispatchQueue.main.async {
            if self._state != newState {
                print("⚙️ \(self.scriptName) state changing from \(self._state) to \(newState)")
                self._state = newState
            }
        }
    }

    private func isErrorState(_ state: LLMEngine.EngineState) -> Bool {
        if case .error(_) = state { return true }
        return false
    }

    private func setupStdErrHandler() {
        guard let errPipe = stderrPipe else {
            print("❌ Cannot setup stderr handler for \(scriptName): pipe is nil.")
            return
        }
        let errHandle = errPipe.fileHandleForReading
        errHandle.readabilityHandler = { [weak self] pipe in
            guard let self = self else { return }
            let data = pipe.availableData
            if data.isEmpty {
                print("🐍 \(self.scriptName) STDERR: EOF reached or pipe closed.")
                if !(self.task?.isRunning ?? false) { return }
            } else {
                let rawOutput = String(data: data, encoding: .utf8) ?? "<failed to decode stderr as utf8>"
                DispatchQueue.main.async {
                    print("🐍 \(self.scriptName) STDERR: \"\(rawOutput.trimmingCharacters(in: .whitespacesAndNewlines))\"")
                }
            }
            if self.task?.isRunning ?? false {
                pipe.waitForDataInBackgroundAndNotify()
            }
        }
        print("🔧 \(scriptName) STDERR handler setup.")
    }

    private func setupStdOutHandler() {
        guard let outPipe = stdoutPipe else {
            print("❌ Cannot setup stdout handler for \(scriptName): pipe is nil.")
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
                    outputBuffer = lines.removeLast()

                    for line in lines where !line.isEmpty {
                        DispatchQueue.main.async {
                            if self.state == .starting && line == "READY" {
                                print("✅ \(self.scriptName) is ready.")
                                self.updateState(.running)
                                self.isAbortedManually = false
                            } else if self.state == .running && !self.isAbortedManually {
                                if line == "STREAM" {
                                    self.accumulatedOutput = ""
                                } else if line == "END" {
                                    if let callback = self.currentCompletionCallback {
                                        callback(.success(self.accumulatedOutput))
                                    }
                                    self.currentTokenCallback = nil
                                    self.currentCompletionCallback = nil
                                    self.accumulatedOutput = ""
                                    self.isAbortedManually = false
                                } else {
                                    if let callback = self.currentTokenCallback {
                                        callback(line)
                                        self.accumulatedOutput += line + "\n"
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
        print("🔧 \(scriptName) STDOUT handler setup.")
    }

    private func setupTerminationHandler() {
        task?.terminationHandler = { [weak self] process in
            guard let self = self else { return }
            DispatchQueue.main.async {
                print("🔄 \(self.scriptName) process terminated. PID: \(process.processIdentifier), Exit Code: \(process.terminationStatus)")
                let previousState = self.state
                if previousState != .stopped && !self.isAbortedManually {
                    let errorMsg = "\(self.scriptName) terminated unexpectedly (Exit code: \(process.terminationStatus))."
                    print("‼️ \(errorMsg)")
                    self.updateState(.error(errorMsg))
                    if let callback = self.currentCompletionCallback {
                        callback(.failure(.scriptError(errorMsg)))
                    }
                } else {
                    print("ℹ️ \(self.scriptName) terminated normally.")
                    self.updateState(.stopped)
                }
                self.currentTokenCallback = nil
                self.currentCompletionCallback = nil
                self.accumulatedOutput = ""
                self.task = nil
            }
        }
        print("🔧 \(scriptName) termination handler setup.")
    }
    
    func sendCommand(_ command: String) {
        guard state == .running else {
            print("❌ \(scriptName) not running, cannot send command: \(command)")
            return
        }
        guard let stdin = stdinPipe else {
            print("❌ Stdin pipe not available for \(scriptName)")
            return
        }
        guard let inputData = ("CMD:\(command)\n").data(using: .utf8) else {
            print("❌ Error encoding command to UTF-8: \(command)")
            return
        }
        let stdinHandle = stdin.fileHandleForWriting
        do {
            print("➡️ Sending command to \(scriptName): \"\(command)\"")
            if #available(macOS 10.15.4, *) {
                try stdinHandle.write(contentsOf: inputData)
            } else {
                stdinHandle.write(inputData)
            }
        } catch {
            print("🫩 Error writing command to \(scriptName) stdin: \(error)")
        }
    }
    
    private func loadHFToken() -> String? {
        guard let resourcesPath = Bundle.main.resourcePath else { return nil }
        let configPath = resourcesPath + "/config.env"
        
        do {
            let configContent = try String(contentsOfFile: configPath, encoding: .utf8)
            for line in configContent.components(separatedBy: .newlines) {
                let trimmedLine = line.trimmingCharacters(in: .whitespacesAndNewlines)
                if trimmedLine.hasPrefix("HF_TOKEN=") {
                    let token = String(trimmedLine.dropFirst("HF_TOKEN=".count))
                    return token.isEmpty ? nil : token
                }
            }
        } catch {
            print("⚠️ Failed to load config.env: \(error)")
        }
        return nil
    }
    
}

class AudioScriptRunner: BaseScriptRunner {
    init(modelDirectory: String?) {
        super.init(scriptName: "audio.py", modelDirectory: modelDirectory)
    }
}

class AutocompleteScriptRunner: BaseScriptRunner {
    init(modelDirectory: String?) {
        super.init(scriptName: "autocomplete.py", modelDirectory: modelDirectory)
    }
}

class CaretScriptRunner: BaseScriptRunner {
    init(modelDirectory: String?) {
        super.init(scriptName: "embedding_processor.py", modelDirectory: modelDirectory)
    }
}

class EmbeddingsRunner: BaseScriptRunner {
    init(modelDirectory: String?) {
        super.init(scriptName: "text_to_embeddings.py", modelDirectory: modelDirectory)
    }
}

class AttentionRunner: BaseScriptRunner {
    init(modelDirectory: String?) {
        super.init(scriptName: "attention.py", modelDirectory: modelDirectory)
    }
}

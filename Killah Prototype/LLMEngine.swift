//
//  LLMEngine.swift
//  Killah Prototype
//
//  Created by Владислав Калиниченко on 03.05.2025.
//
import Foundation
import Combine
import AppKit

// Define a custom error for LLM operations
enum LLMError: Error, Equatable {
    case engineNotRunning
    case pythonScriptNotReady
    case promptEncodingError
    case stdinWriteError(String)
    case processLaunchError(String)
    case scriptError(String)
    case aborted
    case unknown
}

class LLMEngine: ObservableObject {
    @Published var suggestion: String = "" 
    @Published var engineState: EngineState = .idle

    var task: Process?
    private var stdinPipe: Pipe?
    private var stdoutPipe: Pipe?
    private var stderrPipe: Pipe?
    private var cancellables = Set<AnyCancellable>()
    private var lastSentPrompt: String?

    // Callbacks for the current suggestion generation
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

        // Find Python binary - directly in Resources/venv/bin/python3
        guard let resourcesPath = Bundle.main.resourcePath else {
            updateEngineState(.error("Bundle resources path not found"))
            return
        }
        
        let venvPythonPath = resourcesPath + "/venv/bin/python3"
        guard FileManager.default.fileExists(atPath: venvPythonPath) else {
            print("❌ Python binary not found at: \(venvPythonPath)")
            print("📁 Resources directory contents:")
            if let contents = try? FileManager.default.contentsOfDirectory(atPath: resourcesPath) {
                for item in contents {
                    print("   - \(item)")
                }
            }
            updateEngineState(.error("Python binary not found"))
            return
        }
        print("� Python interpreter path: \(venvPythonPath)")

        // Find Python script - directly in Resources/autocomplete.py
        let scriptPath = resourcesPath + "/autocomplete.py"
        guard FileManager.default.fileExists(atPath: scriptPath) else {
            print("❌ Python script not found at: \(scriptPath)")
            updateEngineState(.error("autocomplete.py not found"))
            return
        }
        print("📜 Python script path: \(scriptPath)")
            
        // Check for model file - directly in Resources/minillm_export.pt
        let modelPath = resourcesPath + "/minillm_export.pt"
        if FileManager.default.fileExists(atPath: modelPath) {
            print("📊 Model file found at: \(modelPath)")
        } else {
            print("⚠️ Model file not found at: \(modelPath)")
            print("   Script might fail if model is required.")
        }

        process.executableURL = URL(fileURLWithPath: venvPythonPath)
        process.arguments = [scriptPath]

        print("🔧 Setting up pipes...")
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
            print("✅ Python process launched. PID: \(process.processIdentifier). Waiting for ready signal from script...")
        } catch {
            print("🫩 Error launching Python process: \(error)")
            updateEngineState(.error("Launch fail: \(error.localizedDescription)"))
            self.task = nil
            currentTokenCallback = nil
            currentCompletionCallback?(.failure(.processLaunchError(error.localizedDescription)))
            currentCompletionCallback = nil
        }
    }
    
    // Helper to check if current engineState is any .error state
    private func isErrorState(_ state: EngineState) -> Bool {
        if case .error(_) = state {
            return true
        }
        return false
    }

    // Helper method to find resources with better error reporting
    private func findResourcePath(_ resourceName: String, ofType type: String? = nil) -> String? {
        print("🔍 ПОИСК РЕСУРСА: '\(resourceName)' типа '\(type ?? "nil")'")
        
        // Try different possible locations
        let possiblePaths = [
            "Resources/\(resourceName)" + (type != nil ? ".\(type!)" : ""),
            resourceName + (type != nil ? ".\(type!)" : "")
        ]
        
        print("🔍 Проверяем пути:")
        for (index, path) in possiblePaths.enumerated() {
            print("   \(index + 1). '\(path)'")
            if let foundPath = Bundle.main.path(forResource: path, ofType: nil) {
                print("✅ НАЙДЕН РЕСУРС: \(foundPath)")
                
                // Проверяем, что файл действительно существует
                if FileManager.default.fileExists(atPath: foundPath) {
                    print("✅ ФАЙЛ ФИЗИЧЕСКИ СУЩЕСТВУЕТ: \(foundPath)")
                    return foundPath
                } else {
                    print("❌ ФАЙЛ НЕ СУЩЕСТВУЕТ ФИЗИЧЕСКИ: \(foundPath)")
                }
            } else {
                print("❌ НЕ НАЙДЕН: '\(path)'")
            }
        }
        
        // МАКСИМАЛЬНО ПОДРОБНАЯ отладочная информация
        print("❌ РЕСУРС '\(resourceName)' НЕ НАЙДЕН НИГДЕ!")
        
        if let resourceURL = Bundle.main.resourceURL {
            print("📁 ПАПКА BUNDLE RESOURCES: \(resourceURL.path)")
            
            do {
                let contents = try FileManager.default.contentsOfDirectory(atPath: resourceURL.path)
                print("📄 ВСЕ РЕСУРСЫ В BUNDLE (\(contents.count) штук):")
                for (index, item) in contents.enumerated() {
                    let itemPath = resourceURL.appendingPathComponent(item).path
                    var isDirectory: ObjCBool = false
                    let exists = FileManager.default.fileExists(atPath: itemPath, isDirectory: &isDirectory)
                    let type = isDirectory.boolValue ? "📁 ПАПКА" : "📄 ФАЙЛ"
                    print("   \(index + 1). \(type): '\(item)' (exists: \(exists))")
                    
                    // Если это папка, заглянем внутрь
                    if isDirectory.boolValue && item.lowercased().contains("venv") {
                        print("     🔍 СОДЕРЖИМОЕ ПАПКИ venv:")
                        do {
                            let venvContents = try FileManager.default.contentsOfDirectory(atPath: itemPath)
                            for venvItem in venvContents {
                                print("       - \(venvItem)")
                                
                                // Если это bin, заглянем и туда
                                if venvItem == "bin" {
                                    let binPath = itemPath + "/bin"
                                    print("         � СОДЕРЖИМОЕ bin:")
                                    do {
                                        let binContents = try FileManager.default.contentsOfDirectory(atPath: binPath)
                                        for binItem in binContents {
                                            print("           - \(binItem)")
                                        }
                                    } catch {
                                        print("           ❌ ОШИБКА ЧТЕНИЯ bin: \(error)")
                                    }
                                }
                            }
                        } catch {
                            print("     ❌ ОШИБКА ЧТЕНИЯ venv: \(error)")
                        }
                    }
                }
            } catch {
                print("❌ НЕ МОГУ ПРОЧИТАТЬ СОДЕРЖИМОЕ BUNDLE: \(error)")
            }
        } else {
            print("❌ BUNDLE RESOURCE URL НЕ НАЙДЕН!")
        }
        
        // Также проверим основные пути приложения
        print("📱 ИНФОРМАЦИЯ О ПРИЛОЖЕНИИ:")
        print("   Bundle path: \(Bundle.main.bundlePath)")
        print("   Resource path: \(Bundle.main.resourcePath ?? "nil")")
        print("   Executable path: \(Bundle.main.executablePath ?? "nil")")
        
        return nil
    }

    // New generateSuggestion with callbacks
    func generateSuggestion(
        prompt: String,
        tokenStreamCallback: @escaping (String) -> Void,
        onComplete: @escaping (Result<String, LLMError>) -> Void
    ) {
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

        if self.currentTokenCallback != nil || self.currentCompletionCallback != nil {
            print("ℹ️ Aborting previous suggestion to start new one for prompt: \"\(prompt)\"")
            self.abortCurrentSuggestion(notifyPython: false)
        }
        
        self.lastSentPrompt = prompt
        self.currentTokenCallback = tokenStreamCallback
        self.currentCompletionCallback = onComplete
        self.accumulatedSuggestion = ""
        self.isAbortedManually = false
        
        print("➡️ Sending prompt to Python: \"\(prompt)\"")
        guard let data = (prompt + "\n").data(using: .utf8) else {
            print("❌ Error encoding prompt to UTF-8 data.")
            currentCompletionCallback?(.failure(.promptEncodingError))
            self.currentTokenCallback = nil
            self.currentCompletionCallback = nil
            return
        }
        
        let stdinHandle = stdin.fileHandleForWriting
        do {
            // Ensure the handle hasn't been closed due to a previous error or stop
            if #available(macOS 10.15.4, *) {
                 try stdinHandle.write(contentsOf: data)
            } else {
                 stdinHandle.write(data) // Fallback for older macOS versions
            }
        } catch {
            print("🫩 Error writing to Python stdin: \(error)")
            engineState = .error("Error writing to Python: \(error.localizedDescription)")
            currentCompletionCallback?(.failure(.stdinWriteError(error.localizedDescription)))
            self.currentTokenCallback = nil
            self.currentCompletionCallback = nil
            // Consider stopping the engine or attempting recovery
        }
    }

    func abortCurrentSuggestion(notifyPython: Bool = true) {
        print("ℹ️ Aborting current suggestion stream. Notify Python: \(notifyPython). Current state: \(engineState)")
        isAbortedManually = true

        if notifyPython, let runningTask = task, runningTask.isRunning, let stdin = stdinPipe {
            // Send an empty line to Python to signal interruption of the current stream
            // The Python script is designed to interpret an empty line as an interruption signal.
            print("➡️ Sending abort signal (empty line) to Python stdin.")
            guard let data = "\n".data(using: .utf8) else {
                print("❌ Error encoding abort signal (empty line) to UTF-8 data.")
                // Proceed with Swift-side cleanup even if we can't notify Python
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
                print("🫩 Error writing abort signal (empty line) to Python stdin: \(error)")
                currentCompletionCallback?(.failure(.stdinWriteError(error.localizedDescription)))
            }
        } else {
            print("ℹ️ Cannot notify Python to abort: task not running or stdin not available.")
        }

        // Call completion callback if it exists, indicating abortion
        if let callback = currentCompletionCallback {
            callback(.failure(.aborted))
        }

        // Clear callbacks and reset state
        currentTokenCallback = nil
        currentCompletionCallback = nil
        accumulatedSuggestion = ""
        // Do not reset isAbortedManually here, setupStdOutHandler might need it
        // to differentiate between manual abort and Python finishing normally.
    }
    
    func stopEngine() {
        print("🛑 Attempting to stop Python engine. Current state: \(engineState)")

        if currentTokenCallback != nil || currentCompletionCallback != nil {
            print("ℹ️ Active suggestion generation during stopEngine. Aborting it.")
            abortCurrentSuggestion(notifyPython: true)
        }
        
        cancellables.forEach { $0.cancel() } // Cancel all Combine subscriptions
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
        } else {
            print("ℹ️ No running Python process to terminate, or task already nil.")
        }
        task = nil

        print("🚽 Cleaning up pipes...")
        stdoutPipe?.fileHandleForReading.readabilityHandler = nil
        stderrPipe?.fileHandleForReading.readabilityHandler = nil
        
        stdinPipe = nil
        stdoutPipe = nil
        stderrPipe = nil

        updateEngineState(.stopped)
        DispatchQueue.main.async {
            self.suggestion = "" // Clear any old suggestion displayed via @Published var
        }
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
                // If task is not running, no more data will come.
                if !(self.task?.isRunning ?? false) {
                    return
                }
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
            
            // Re-register for data availability notifications if the task is still running
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
            if data.isEmpty {
                print("🐍 STDOUT: EOF reached. Task running: \(self.task?.isRunning ?? false). State: \(self.engineState)")
                if self.engineState == .running || self.engineState == .starting {
                    if !self.isAbortedManually, let callback = self.currentCompletionCallback {
                        DispatchQueue.main.async {
                            print("‼️ STDOUT EOF: Python script exited prematurely or stdout closed unexpectedly.")
                            callback(.failure(.scriptError("Python script exited prematurely or stdout closed.")))
                            self.currentTokenCallback = nil
                            self.currentCompletionCallback = nil
                            self.updateEngineState(.error("Python script exited prematurely"))
                        }
                    }
                }
                // If task is not running, no more data will come.
                if !(self.task?.isRunning ?? false) {
                    return
                }
            }

            // Only process if data is not empty
            if !data.isEmpty {
                let rawOutput = String(data: data, encoding: .utf8) ?? "<failed to decode stdout as utf8>"

                let lines = rawOutput.components(separatedBy: .newlines)
                for lineContent in lines {
                    let line = lineContent.trimmingCharacters(in: .whitespacesAndNewlines)
                    if line.isEmpty { continue }

                    print("🐍 STDOUT Line: [\(line)]")

                    DispatchQueue.main.async {
                        if self.engineState == .starting {
                            if line == "Entering main processing loop." {
                                print("✅ Python script is ready and processing loop entered.")
                                self.updateEngineState(.running)
                                self.isAbortedManually = false
                            } else {
                                print("⚠️ Unexpected output from Python during startup: [\(line)]")
                            }
                        }

                        if self.engineState != .running && line != "Entering main processing loop." {
                            print("ℹ️ STDOUT: Ignoring line [\(line)] as engine is not in .running state (current: \(self.engineState))")
                            return
                        }
                        
                        if self.isAbortedManually {
                            print("ℹ️ Ignoring token [\(line)] because suggestion was aborted.")
                            return
                        }

                        if line == "Streaming suggestions..." {
                            self.accumulatedSuggestion = ""
                            return 
                        } else if line == "END_SUGGESTIONS" {
                            if let callback = self.currentCompletionCallback {
                                callback(.success(self.accumulatedSuggestion))
                            }
                            self.currentTokenCallback = nil
                            self.currentCompletionCallback = nil
                            self.accumulatedSuggestion = ""
                            self.isAbortedManually = false
                            return
                        } else if line != "Entering main processing loop." { 
                            if let callback = self.currentTokenCallback {
                                callback(line)
                                self.accumulatedSuggestion += line
                            }
                        }
                    }
                }
            }
            
            // Re-register for data availability notifications if the task is still running
            if self.task?.isRunning ?? false {
                 pipe.waitForDataInBackgroundAndNotify() // 'pipe' is the FileHandle (outHandle)
            }
        }
        
        // Initial call to start listening for data
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
                    // If it was running/starting and terminated (e.g., by stopEngine() or script exiting cleanly after stdin close)
                    print("ℹ️ Python process terminated, likely as part of normal stop or script completion.")
                    self.updateEngineState(.stopped) // Ensure state is stopped
                }
                
                self.currentTokenCallback = nil
                self.currentCompletionCallback = nil
                self.accumulatedSuggestion = ""
                // self.isAbortedManually = false // Reset by startEngine or when a suggestion completes normally
                
                // Task is nilled by stopEngine or if launch fails. Here we just react.
                // If task is not nil here, it means termination happened outside of stopEngine flow.
                if self.task != nil { // If task still exists, it wasn't stopped by stopEngine
                    self.task = nil // Clean up our reference if termination handler is called directly
                }
            }
        }
        print("🔧 Process termination handler setup.")
    }
    
    // Removed the old abort() method, replaced by abortCurrentSuggestion() and stopEngine()
    // func abort() { ... }
    
    deinit {
        print("🗑️ LLMEngine deinit - Stopping engine.")
        stopEngine() // Ensure engine is stopped when LLMEngine instance is deallocated
    }
}

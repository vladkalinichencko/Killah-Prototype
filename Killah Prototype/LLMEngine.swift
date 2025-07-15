import Foundation
import Combine
import AppKit
import CryptoKit // Required for CacheManager's SHA-256 extension

class LLMEngine: ObservableObject {
    @Published var suggestion: String = ""
    @Published var engineState: EngineState = .idle
    
    enum EngineState: Equatable {
        case idle
        case starting
        case running
        case stopped
        case error(String)
        static func == (lhs: EngineState, rhs: EngineState) -> Bool {
            switch (lhs, rhs) {
            case (.idle, .idle), (.starting, .starting), (.running, .running), (.stopped, .stopped):
                return true
            case let (.error(lhsError), .error(rhsError)):
                return lhsError == rhsError
            default:
                return false
            }
        }
    }

    enum LLMError: Error {
        case engineNotRunning
        case pythonScriptNotReady
        case processLaunchError(String)
        case promptEncodingError
        case stdinWriteError(String)
        case scriptError(String)
        case aborted
    }

    private var runners: [String: PythonScriptRunner] = [:]
    private var modelServer: ModelServerRunner
    private var cancellables = Set<AnyCancellable>()
    private var currentTemperature: Float = 0.8 // Начальное значение температуры
    
    init(modelManager: ModelManager) {
        print("LLMEngine init")
        let modelDir = modelManager.getModelsDirectory().path
        
        // Initialize the model server
        modelServer = ModelServerRunner(modelDirectory: modelDir)
        modelServer.start() // Start the server first
        
        // Initialize Python script runners
        runners["audio"] = AudioScriptRunner(modelDirectory: modelDir)
        runners["autocomplete"] = AutocompleteScriptRunner(modelDirectory: modelDir)
        runners["embeddings"] = EmbeddingsRunner(modelDirectory: modelDir)
        runners["caret"] = CaretScriptRunner(modelDirectory: modelDir)

        NotificationCenter.default.publisher(for: NSApplication.willTerminateNotification)
            .sink { [weak self] _ in
                print("App is terminating, stopping engine...")
                self?.stopEngine()
            }
            .store(in: &cancellables)
    }
    
    func getRunnerState(for key: String) -> EngineState? {
            return runners[key]?.state
        }

    func startEngine(for script: String) {
        guard let runner = runners[script] else {
            print("❌ Unknown script: \(script)")
            updateEngineState(.error("Unknown script: \(script)"))
            return
        }
        // Invalidate cache when starting engine, as model may change
        CacheManager.shared.invalidateCache()
        runner.start()
        updateEngineState(runner.state)
    }

    func generateSuggestion(
        for script: String,
        prompt: String,
        loraAdapter: String? = nil, // Новый параметр для LoRA
        tokenStreamCallback: @escaping (String) -> Void,
        onComplete: @escaping (Result<String, LLMError>) -> Void
    ) {
        // Check cache first
        if let cachedSuggestion = CacheManager.shared.getCachedSuggestion(for: prompt, temperature: self.currentTemperature) {
            print("📦 Cache hit for prompt: \"\(prompt)\"")
            // Split cached suggestion into tokens like in BaseScriptRunner
            let tokens = cachedSuggestion.components(separatedBy: .newlines).filter { !$0.isEmpty }
            // Send tokens with a delay to mimic streaming
            var currentIndex = 0
            func sendNextToken() {
                guard currentIndex < tokens.count else {
                    onComplete(.success(cachedSuggestion))
                    return
                }
                let token = tokens[currentIndex]
                print("📦 Sending cached token: \"\(token)\"")
                tokenStreamCallback(token)
                currentIndex += 1
                DispatchQueue.main.asyncAfter(deadline: .now() + 0.2) {
                    sendNextToken()
                }
            }
            sendNextToken()
            return
        }
        
        guard let runner = runners[script] else {
            print("❌ Unknown script: \(script)")
            onComplete(.failure(.scriptError("Unknown script: \(script)")))
            return
        }
        print("📄 Generating suggestion for \(script) with prompt: \"\(prompt)\"")
        // Применяем LoRA-адаптер перед отправкой промпта, если он указан
        if script == "autocomplete", let loraAdapter = loraAdapter {
            modelServer.applyLoraAdapter(adapterName: loraAdapter) { result in
                switch result {
                case .success:
                    print("📄 Generating suggestion for \(script) with prompt: \"\(prompt)\" using LoRA: \(loraAdapter)")
                    runner.sendData(prompt, tokenStreamCallback: tokenStreamCallback,
                        onComplete: { result in
                        switch result {
                        case .success(let suggestion):
                            print("Saving suggestion to cache for prompt \"\(prompt)\"")
                            CacheManager.shared.setCachedSuggestion(suggestion, for: prompt, temperature: self.currentTemperature)
                            onComplete(.success(suggestion))
                        case .failure(let error):
                            onComplete(.failure(error))
                        }
                    })
                case .failure(let error):
                    print("🫩 Failed to apply LoRA adapter: \(error)")
                    onComplete(.failure(.scriptError("Failed to apply LoRA adapter: \(error.localizedDescription)")))
                }
            }
        } else {
            print("📄 Generating suggestion for \(script) with prompt: \"\(prompt)\"")
            runner.sendData(prompt, tokenStreamCallback: tokenStreamCallback,
                onComplete: { result in
                switch result {
                case .success(let suggestion):
                    print("Saving suggestion to cache for prompt \"\(prompt)\"")
                    CacheManager.shared.setCachedSuggestion(suggestion, for: prompt, temperature: self.currentTemperature)
                    onComplete(.success(suggestion))
                case .failure(let error):
                    onComplete(.failure(error))
                }
            })
        }
        updateEngineState(runner.state)
    }
    
    func sendCommand(_ command: String, for script: String) {
        guard let runner = runners[script] else {
            print("❌ Unknown script: \(script)")
            return
        }
        
        if command == "INCREASE_TEMPERATURE" {
            currentTemperature = min(currentTemperature + 0.1, 2.0)
            print("🌡️ Temperature increased to \(currentTemperature)")
        } else if command == "DECREASE_TEMPERATURE" {
            currentTemperature = max(currentTemperature - 0.1, 0.1)
            print("🌡️ Temperature decreased to \(currentTemperature)")
        }
        
        runner.sendCommand(command)
    }
    
    func stopEngine(for script: String? = nil) {
        if let script = script, let runner = runners[script] {
            runner.stop()
            updateEngineState(runner.state)
        } else {
            modelServer.stop()
            runners.forEach { $0.value.stop() }
            updateEngineState(.stopped)
        }
    }
    
    func abortSuggestion(for script: String, notifyPython: Bool = true) {
        guard let runner = runners[script] else {
            print("❌ Unknown script: \(script)")
            return
        }
        print("ℹ️ Aborting suggestion for \(script)")
        runner.abortSuggestion(notifyPython: notifyPython)
        updateEngineState(runner.state)
    }
    
    private func updateEngineState(_ newState: EngineState) {
        DispatchQueue.main.async {
            if self.engineState != newState {
                print("⚙️ LLMEngine state changing from \(self.engineState) to \(newState)")
                self.engineState = newState
            }
        }
    }

    deinit {
        print("🗑️ LLMEngine deinit - Stopping engine.")
        print(Thread.callStackSymbols.joined(separator: "\n"))
        stopEngine()
    }
}

class ModelServerRunner {
    private var serverProcess: Process?
    private var _state: LLMEngine.EngineState = .idle
    private let modelDirectory: String
    private let loraAdapters: [String] // Список LoRA-адаптеров

    init(modelDirectory: String) {
        self.modelDirectory = modelDirectory
        let appSupportDir = FileManager.default.urls(for: .applicationSupportDirectory, in: .userDomainMask).first!
                    .appendingPathComponent("KillahPrototype/models/lora").path
        self.loraAdapters = [
                "\(appSupportDir)/autocomplete_lora.gguf"
                ] // Список всех LoRA-адаптеров
    }

    var state: LLMEngine.EngineState {
        return _state
    }

    func start() {
        guard state == .idle || state == .stopped else {
            print("ℹ️ Model server already running or starting")
            return
        }
        
        print("🚀 Starting llama-server...")
        updateState(.starting)
        
        let process = Process()
        serverProcess = process
        
        guard let resourcesPath = Bundle.main.resourcePath else {
            updateState(.error("Bundle resources path not found"))
            return
        }
        
        let serverPath = resourcesPath + "/venv/bin/llama-server"
        let modelPath = modelDirectory + "/gemma/gemma-3-4b-pt-q4_0.gguf"
        
        var arguments = [
            "-m", modelPath,
            "--port", "8080",
            "--host", "localhost",
            "--n-gpu-layers", "1",
            "--embedding",
            "--lora-init-without-apply" // Загружаем адаптеры без применения
        ]
        
        // Добавляем все LoRA-адаптеры
        for loraPath in loraAdapters {
            arguments.append("--lora")
            arguments.append(loraPath)
        }
        
        process.executableURL = URL(fileURLWithPath: serverPath)
        process.arguments = arguments
        
        let stderrPipe = Pipe()
        process.standardError = stderrPipe
        
        stderrPipe.fileHandleForReading.readabilityHandler = { pipe in
            let data = pipe.availableData
            if !data.isEmpty, let output = String(data: data, encoding: .utf8) {
                print("🐍 llama-server STDERR: \"\(output.trimmingCharacters(in: .whitespacesAndNewlines))\"")
                if output.contains("llama server listening") {
                    self.updateState(.running)
                }
            }
        }
        
        do {
            try process.run()
            print("✅ llama-server launched. PID: \(process.processIdentifier)")
        } catch {
            print("🫩 Error launching llama-server: \(error)")
            updateState(.error("Launch fail: \(error.localizedDescription)"))
            serverProcess = nil
        }
    }

    func stop() {
        if let process = serverProcess, process.isRunning {
            process.terminate()
            print("🛑 llama-server stopped")
        }
        serverProcess = nil
        updateState(.stopped)
    }

    // Функция для применения LoRA-адаптера через API
    func applyLoraAdapter(adapterName: String, scale: Float = 1.0, completion: @escaping (Result<Void, Error>) -> Void) {
        let url = URL(string: "http://localhost:8080/lora-adapters")!
        var request = URLRequest(url: url)
        request.httpMethod = "POST"
        request.setValue("application/json", forHTTPHeaderField: "Content-Type")
        
        let body: [String: Any] = [
            "adapters": [
                [
                    "path": adapterName, // Например, "lora/autocomplete_lora.gguf"
                    "scale": scale
                ]
            ]
        ]
        
        do {
            request.httpBody = try JSONSerialization.data(withJSONObject: body)
        } catch {
            completion(.failure(error))
            return
        }
        
        URLSession.shared.dataTask(with: request) { data, response, error in
            if let error = error {
                print("🫩 Error applying LoRA adapter: \(error)")
                completion(.failure(error))
                return
            }
            
            if let httpResponse = response as? HTTPURLResponse, httpResponse.statusCode == 200 {
                print("✅ Applied LoRA adapter: \(adapterName)")
                completion(.success(()))
            } else {
                let error = NSError(domain: "", code: -1, userInfo: [NSLocalizedDescriptionKey: "Failed to apply LoRA adapter"])
                completion(.failure(error))
            }
        }.resume()
    }
    
    private func updateState(_ newState: LLMEngine.EngineState) {
        if _state != newState {
            print("⚙️ ModelServerRunner state changing from \(_state) to \(newState)")
            _state = newState
        }
    }
}

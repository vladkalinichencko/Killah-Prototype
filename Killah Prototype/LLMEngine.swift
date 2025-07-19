import Foundation
import Combine
import AppKit
import CryptoKit // Required for CacheManager's SHA-256 extension
import Darwin
import SwiftData

// Структура для декодирования JSON из text_to_embeddings.py
struct EmbeddingResponse: Codable {
    let type: String
    let embeddings: [Float]
}

class LLMEngine: ObservableObject {
    @Published var suggestion: String = ""
    @Published var engineState: EngineState = .idle
    private var activeTasks: [String: Bool] = [:] // Отслеживание активных задач
    
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
    private let modelContainer: ModelContainer
    
    init(modelManager: ModelManager,modelContainer: ModelContainer) {
        print("LLMEngine init")
        self.modelContainer = modelContainer
        let modelDir = modelManager.getModelsDirectory().path
        
        // Initialize the model server
        modelServer = ModelServerRunner(modelDirectory: modelDir)
        modelServer.start() // Start the server first
        
        // Initialize Python script runners
        runners["audio"] = AudioScriptRunner(modelDirectory: modelDir)
        runners["autocomplete"] = AutocompleteScriptRunner(modelDirectory: modelDir)
        runners["embeddings"] = EmbeddingsRunner(modelDirectory: modelDir)
        runners["caret"] = CaretScriptRunner(modelDirectory: modelDir)
        runners["attention"] = AttentionRunner(modelDirectory: modelDir)

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
        isFromCaret: Bool = false,  // Флаг, указывающий, что промпт содержит эмбеддинги
        loraAdapter: String? = nil,
        tokenStreamCallback: @escaping (String) -> Void,
        onComplete: @escaping (Result<String, LLMError>) -> Void
    ) {
        guard let runner = runners[script] else {
            print("❌ Unknown script: \(script)")
            onComplete(.failure(.scriptError("Unknown script: \(script)")))
            return
        }
        if activeTasks[script] == true {
            print("⏳ Waiting for previous task in \(script) to complete")
            runner.abortSuggestion(notifyPython: true) // Прерываем предыдущую задачу
        }
        activeTasks[script] = true
        Task{
            let personalizedDocs = await getPersonalizedDocuments()
            let docEmbeddings = personalizedDocs.map { $0.embedding }
            let docURLs = personalizedDocs.map { $0.url }
            
            if !docEmbeddings.isEmpty {
                if isFromCaret {
                    // Извлекаем эмбеддинги из промпта
                    let embeddingsJson = extractEmbeddingsJson(from: prompt)
                    if let embeddings = parseEmbeddings(from: embeddingsJson) {
                        computeAttentionWeights(target: embeddings, histories: docEmbeddings) { result in
                            switch result {
                            case .success(let weights):
                                print("ℹ️ Веса внимания: \(weights)")
                                let threshold = 0.5
                                var selectedEmbeddings: [URL] = []
                                for (index, weight) in weights.enumerated() where weight > threshold {
                                    let embedURL = docURLs[index].deletingPathExtension().appendingPathExtension("pt")
                                    selectedEmbeddings.append(embedURL)
                                }
                                print("ℹ️ Выбрано эмбеддингов: \(selectedEmbeddings.count)")
                                
                                var augmentedPrompt = prompt
                                if !selectedEmbeddings.isEmpty {
                                    augmentedPrompt += "[Контекст из персонализированных документов]"
                                }
                                
                                self.continueGeneration(
                                    script: script,
                                    prompt: augmentedPrompt.replacingOccurrences(of: "\n", with: " "),
                                    loraAdapter: loraAdapter,
                                    tokenStreamCallback: tokenStreamCallback,
                                    onComplete: onComplete
                                )
                            case .failure(let error):
                                print("🫩 Ошибка вычисления весов внимания: \(error)")
                                self.continueGeneration(
                                    script: script,
                                    prompt: prompt,
                                    loraAdapter: loraAdapter,
                                    tokenStreamCallback: tokenStreamCallback,
                                    onComplete: onComplete
                                )
                            }
                        }
                    } else {
                        print("🫩 Ошибка: не удалось извлечь или распарсить эмбеддинги из промпта")
                        continueGeneration(
                            script: script,
                            prompt: prompt,
                            loraAdapter: loraAdapter,
                            tokenStreamCallback: tokenStreamCallback,
                            onComplete: onComplete
                        )
                    }
                } else {
                    // Generate embedding for the prompt
                    generateEmbedding(for: prompt) { [weak self] result in
                        guard let self = self else { return }
                        switch result {
                        case .success(let targetEmbedding):
                            self.computeAttentionWeights(target: targetEmbedding, histories: docEmbeddings) { result in
                                switch result {
                                case .success(let weights):
                                    print("ℹ️ Веса внимания: \(weights)")
                                    let threshold = 0.5
                                    var selectedEmbeddings: [URL] = []
                                    for (index, weight) in weights.enumerated() where weight > threshold {
                                        let embedURL = docURLs[index].deletingPathExtension().appendingPathExtension("pt")
                                        selectedEmbeddings.append(embedURL)
                                    }
                                    print("ℹ️ Выбрано эмбеддингов: \(selectedEmbeddings.count)")
                                    
                                    var augmentedPrompt = prompt
                                    if !selectedEmbeddings.isEmpty {
                                        augmentedPrompt += "[Контекст из персонализированных документов]"
                                    }
                                    
                                    self.continueGeneration(
                                        script: script,
                                        prompt: augmentedPrompt.replacingOccurrences(of: "\n", with: " "),
                                        loraAdapter: loraAdapter,
                                        tokenStreamCallback: tokenStreamCallback,
                                        onComplete: onComplete
                                    )
                                case .failure(let error):
                                    print("🫩 Ошибка вычисления весов внимания: \(error)")
                                    self.continueGeneration(
                                        script: script,
                                        prompt: prompt,
                                        loraAdapter: loraAdapter,
                                        tokenStreamCallback: tokenStreamCallback,
                                        onComplete: onComplete
                                    )
                                }
                            }
                        case .failure(let error):
                            print("🫩 Ошибка генерации эмбеддинга для промпта: \(error)")
                            self.continueGeneration(
                                script: script,
                                prompt: prompt,
                                loraAdapter: loraAdapter,
                                tokenStreamCallback: tokenStreamCallback,
                                onComplete: onComplete
                            )
                        }
                    }
                }
            } else {
                continueGeneration(
                    script: script,
                    prompt: prompt,
                    loraAdapter: loraAdapter,
                    tokenStreamCallback: tokenStreamCallback,
                    onComplete: onComplete
                )
            }
        }
    }
    
    // Вспомогательный метод для продолжения генерации
    private func continueGeneration(
        script: String,
        prompt: String,
        loraAdapter: String?,
        tokenStreamCallback: @escaping (String) -> Void,
        onComplete: @escaping (Result<String, LLMError>) -> Void
    ) {
        if let cachedSuggestion = CacheManager.shared.getCachedSuggestion(for: prompt, temperature: self.currentTemperature) {
            print("📦 Cache hit for prompt: \"\(prompt)\"")
            let tokens = cachedSuggestion.components(separatedBy: .newlines).filter { !$0.isEmpty }
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
        print("📄 Generating suggestion for \(script) with prompt: \"\(prompt.prefix(100))\"")
        if script == "autocomplete", let loraAdapter = loraAdapter {
            modelServer.applyLoraAdapter(adapterName: loraAdapter) { result in
                switch result {
                case .success:
                    print("📄 Generating suggestion with LoRA: \(loraAdapter)")
                    runner.sendData(prompt, tokenStreamCallback: tokenStreamCallback) { result in
                        self.activeTasks[script] = false // Сбрасываем состояние после завершения
                        switch result {
                        case .success(let suggestion):
                            CacheManager.shared.setCachedSuggestion(suggestion, for: prompt, temperature: self.currentTemperature)
                            onComplete(.success(suggestion))
                        case .failure(let error):
                            onComplete(.failure(error))
                        }
                    }
                case .failure(let error):
                    onComplete(.failure(.scriptError("Failed to apply LoRA adapter: \(error.localizedDescription)")))
                }
            }
        } else {
            runner.sendData(prompt, tokenStreamCallback: tokenStreamCallback) { result in
                self.activeTasks[script] = false // Сбрасываем состояние после завершения
                switch result {
                case .success(let suggestion):
                    CacheManager.shared.setCachedSuggestion(suggestion, for: prompt, temperature: self.currentTemperature)
                    onComplete(.success(suggestion))
                case .failure(let error):
                    onComplete(.failure(error))
                }
            }
        }
        updateEngineState(runner.state)
    }
    
    // Вспомогательные функции для извлечения и парсинга эмбеддингов
    private func extractEmbeddingsJson(from prompt: String) -> String? {
        // Предполагаем, что формат промпта: "embeddingsJson|||prompt_text"
        let parts = prompt.split(separator: "|||", maxSplits: 1)
        if parts.count == 2 {
            return String(parts[0])
        }
        return nil
    }
    
    private func parseEmbeddings(from jsonString: String?) -> [Float]? {
        guard let jsonString = jsonString else { return nil }
        do {
            let data = jsonString.data(using: .utf8)!
            let json = try JSONSerialization.jsonObject(with: data, options: []) as! [String: Any]
            if let embeddings = json["embeddings"] as? [Double] {
                return embeddings.map { Float($0) }
            } else if let embeddings = json["embeddings"] as? [Float] {
                return embeddings
            }
        } catch {
            print("🫩 Ошибка парсинга JSON эмбеддингов: \(error)")
        }
        return nil
    }
    
    func computeAttentionWeights(
        target: [Float],
        histories: [[Float]],
        onComplete: @escaping (Result<[Double], LLMError>) -> Void
    ) {
        guard let runner = runners["attention"] else {
            onComplete(.failure(.scriptError("Attention runner not found")))
            return
        }
        
        let inputData: [String: Any] = [
            "target": target,
            "histories": histories
        ]
        
        do {
            let jsonData = try JSONSerialization.data(withJSONObject: inputData)
            guard let jsonString = String(data: jsonData, encoding: .utf8) else {
                onComplete(.failure(.promptEncodingError))
                return
            }
            
            runner.sendData(jsonString, tokenStreamCallback: { _ in }, onComplete: { result in
                switch result {
                case .success(let output):
                    do {
                        let weights = try JSONDecoder().decode([Double].self, from: Data(output.utf8))
                        onComplete(.success(weights))
                    } catch {
                        onComplete(.failure(.scriptError("Failed to parse attention weights: \(error)")))
                    }
                case .failure(let error):
                    onComplete(.failure(error))
                }
            })
        } catch {
            onComplete(.failure(.promptEncodingError))
        }
    }
    
    func generateEmbedding(
        for text: String,
        onComplete: @escaping (Result<[Float], LLMError>) -> Void
    ) {
        guard let runner = runners["embeddings"] else {
            onComplete(.failure(.scriptError("Embeddings runner not found")))
            return
        }
        let input = text
        runner.sendData(input, tokenStreamCallback: { _ in }) { result in
            switch result {
            case .success(let output):
                do {
                    let data = try JSONDecoder().decode(EmbeddingResponse.self, from: Data(output.utf8))
                    if data.type == "text_embeds" {
                        onComplete(.success(data.embeddings))
                    } else {
                        onComplete(.failure(.scriptError("Invalid embeddings type: \(data.type)")))
                    }
                } catch {
                    onComplete(.failure(.scriptError("Failed to parse embeddings: \(error)")))
                }
            case .failure(let error):
                onComplete(.failure(error))
            }
        }
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

    func getPersonalizedDocuments() async -> [(url: URL, content: String, embedding: [Float])] {
        var personalizedDocs: [(url: URL, content: String, embedding: [Float])] = []
            
        personalizedDocs = await MainActor.run {
            var localDocs: [(url: URL, content: String, embedding: [Float])] = []
            let context = self.modelContainer.mainContext
            print("ℹ️ Using model context: \(ObjectIdentifier(context))")
            do {
                let descriptor = FetchDescriptor<Embedding>(predicate: #Predicate { $0.isPersonalized })
                let embeddings = try context.fetch(descriptor)
                print("ℹ️ Найдено эмбеддингов в базе: \(embeddings.count)")
                for embedding in embeddings {
                    let documentURL = embedding.documentURL.standardizedFileURL
                    if let content = try? String(contentsOf: documentURL, encoding: .utf8),
                       let embeddingArray = try? JSONDecoder().decode([Float].self, from: embedding.embeddingData) {
                        localDocs.append((url: documentURL, content: content, embedding: embeddingArray))
                    } else {
                        print("⚠️ Не удалось загрузить содержимое или эмбеддинг для документа: \(documentURL.lastPathComponent)")
                    }
                }
            } catch {
                print("🫩 Ошибка загрузки персонализированных документов: \(error)")
            }
            return localDocs
        }
        
        print("ℹ️ Найдено персонализированных документов: \(personalizedDocs.count)")
        return personalizedDocs
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
            print("🛑 Останавливаем llama-server с PID: \(process.processIdentifier)")
            process.terminate() // Отправляем SIGTERM
            DispatchQueue.main.asyncAfter(deadline: .now() + 2) {
                if process.isRunning {
                    print("🛑 Сервер llama-server все еще работает, принудительно завершаем")
                    kill(process.processIdentifier, SIGKILL) // Отправляем SIGKILL
                } else {
                    print("🛑 Сервер llama-server успешно остановлен")
                }
            }
        } else {
            print("🛑 Нет запущенного процесса сервера")
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

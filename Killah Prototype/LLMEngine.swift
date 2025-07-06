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
    private var cancellables = Set<AnyCancellable>()

    init(modelManager: ModelManager) {
        print("LLMEngine init")
        let modelDir = modelManager.getModelsDirectory().path
        
        // Инициализация runner'ов для скриптов
        runners["audio"] = AudioScriptRunner(modelDirectory: modelDir)
        runners["autocomplete"] = AutocompleteScriptRunner(modelDirectory: modelDir)

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
        tokenStreamCallback: @escaping (String) -> Void,
        onComplete: @escaping (Result<String, LLMError>) -> Void
    ) {
        // Check cache first
        if let cachedSuggestion = CacheManager.shared.getCachedSuggestion(for: prompt) {
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
        runner.sendData(prompt, tokenStreamCallback: tokenStreamCallback,
            onComplete: { result in
            switch result {
            case .success(let suggestion):
                print("Saving suggestion to cache for prompt \"\(prompt)\"")
                CacheManager.shared.setCachedSuggestion(suggestion, for: prompt)
                onComplete(.success(suggestion))
            case .failure(let error):
                onComplete(.failure(error))
            }
        })
        updateEngineState(runner.state)
    }
    

    func stopEngine(for script: String? = nil) {
        if let script = script, let runner = runners[script] {
            runner.stop()
            updateEngineState(runner.state)
        } else {
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

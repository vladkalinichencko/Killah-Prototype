import Cocoa
import SwiftUI
import Combine // Added for Combine publishers

class AppDelegate: NSObject, NSApplicationDelegate {
    struct Dependencies {
        let llmEngine: LLMEngine
        let audioEngine: AudioEngine
        let themeManager: ThemeManager
        let modelManager: ModelManager
    }

    static var dependencies: Dependencies!

    func applicationDidFinishLaunching(_ notification: Notification) {
        loadEnvironmentVariables()
        performInitialChecks()
        setupDocumentController()
    }
    
    private func setupDocumentController() {
        // Проверяем, есть ли уже сохраненный путь к документам
        if let savedPath = UserDefaults.standard.string(forKey: "DefaultOpenDirectory"), !savedPath.isEmpty {
            // Используем сохраненный путь
            print("📁 Используем сохраненный путь к документам: \(savedPath)")
        } else {
            // Настраиваем папку Killah как папку по умолчанию для открытия файлов
            let documentsURL = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask).first!
            let killahDocumentsURL = documentsURL.appendingPathComponent("Killah")
            
            // Сохраняем путь для использования в диалогах
            UserDefaults.standard.set(killahDocumentsURL.path, forKey: "DefaultOpenDirectory")
            UserDefaults.standard.synchronize()
            print("📁 Установлен путь по умолчанию: \(killahDocumentsURL.path)")
        }
    }
    
    private func loadEnvironmentVariables() {
        guard let resourcesPath = Bundle.main.resourcePath else { return }
        let configPath = resourcesPath + "/config.env"
        
        do {
            let configContent = try String(contentsOfFile: configPath, encoding: .utf8)
            for line in configContent.components(separatedBy: .newlines) {
                let trimmedLine = line.trimmingCharacters(in: .whitespacesAndNewlines)
                if trimmedLine.hasPrefix("HF_TOKEN=") {
                    let token = String(trimmedLine.dropFirst("HF_TOKEN=".count))
                    if !token.isEmpty {
                        setenv("HF_TOKEN", token, 1)
                        print("🔧 Set HF_TOKEN environment variable")
                    }
                }
            }
        } catch {
            print("⚠️ Failed to load config.env: \(error)")
        }
    }
    
    private func performInitialChecks() {
        checkKillahFolder()
        checkModels()
    }
    
    private func checkKillahFolder() {
        // Используем системную папку Documents пользователя
        let documentsURL = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask).first!
        let killahDocumentsURL = documentsURL.appendingPathComponent("Killah")
        
        let fileManager = FileManager.default
        let folderExists = fileManager.fileExists(atPath: killahDocumentsURL.path)
        
        if !folderExists {
            do {
                try fileManager.createDirectory(at: killahDocumentsURL, withIntermediateDirectories: true)
                print("📁 Создана папка Killah в системной папке Documents")
            } catch {
                print("❌ Ошибка создания папки Killah: \(error)")
            }
        }
    }
    
    private func checkModels() {
        guard let modelManager = AppDelegate.dependencies?.modelManager else { return }
        
        // Подписываемся на изменения статуса моделей
        modelManager.$status
            .receive(on: DispatchQueue.main)
            .sink { [weak self] status in
                self?.handleModelStatusChange(status)
            }
            .store(in: &cancellables)
        
        modelManager.verifyModels()
    }
    
    private func handleModelStatusChange(_ status: ModelManager.ModelStatus) {
        switch status {
        case .needsDownloading:
            AppStateManager.shared.isModelDownloading = true
        case .downloading:
            AppStateManager.shared.isModelDownloading = true
        case .ready:
            AppStateManager.shared.isModelDownloading = false
            startPythonScripts()
        case .error:
            AppStateManager.shared.isModelDownloading = false
        case .checking:
            break
        }
    }
    
    private func startPythonScripts() {
        guard let llmEngine = AppDelegate.dependencies?.llmEngine else { return }
        
        AppStateManager.shared.isPythonScriptsStarting = true
        
        llmEngine.startEngine(for: "autocomplete")
        llmEngine.startEngine(for: "audio")
        llmEngine.startEngine(for: "embeddings")
        llmEngine.startEngine(for: "caret")

        
        DispatchQueue.main.asyncAfter(deadline: .now() + 0.5) {
            self.checkPythonScriptsState()
        }
    }
    
    private func checkPythonScriptsState() {
        guard let llmEngine = AppDelegate.dependencies?.llmEngine else { return }
        
        let autocompleteState = llmEngine.getRunnerState(for: "autocomplete")
        let audioState = llmEngine.getRunnerState(for: "audio")
        
        if autocompleteState == .running && audioState == .running {
            AppStateManager.shared.isPythonScriptsStarting = false
        } else if autocompleteState == LLMEngine.EngineState.error("") || audioState == LLMEngine.EngineState.error("") {
            AppStateManager.shared.isPythonScriptsStarting = false
        } else {
            DispatchQueue.main.asyncAfter(deadline: .now() + 0.5) {
                self.checkPythonScriptsState()
            }
        }
    }
    
    private func createFolderIcon(for folderURL: URL) {
        if let appIconPath = Bundle.main.path(forResource: "app-icon-512", ofType: "png", inDirectory: "Assets.xcassets/AppIcon.appiconset") {
            let iconPath = folderURL.appendingPathComponent("folder-icon.png")
            
            do {
                try FileManager.default.copyItem(atPath: appIconPath, toPath: iconPath.path)
            } catch {
                print("❌ Ошибка копирования иконки: \(error)")
            }
        }
    }
    
    private var cancellables = Set<AnyCancellable>()
}

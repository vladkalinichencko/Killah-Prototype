import Cocoa
import SwiftUI

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
        createDocumentsFolder()
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
    
    private func createDocumentsFolder() {
        print("🚀 AppDelegate.createDocumentsFolder() вызвана")
        
        let fileManager = FileManager.default
        print("📂 Получаем путь к Documents...")
        
        // Используем обычную папку Documents пользователя
        let documentsURL = URL(fileURLWithPath: NSHomeDirectory()).appendingPathComponent("Documents")
        print("📂 Documents путь: \(documentsURL.path)")
        
        let killahDocumentsURL = documentsURL.appendingPathComponent("Killah")
        print("📂 Полный путь к папке Killah: \(killahDocumentsURL.path)")
        
        // Проверяем существование папки
        let folderExists = fileManager.fileExists(atPath: killahDocumentsURL.path)
        print("🔍 Папка Killah существует: \(folderExists)")
        
        if !folderExists {
            print("📁 Папка Killah не найдена, создаем...")
            do {
                try fileManager.createDirectory(at: killahDocumentsURL, withIntermediateDirectories: true)
                print("✅ Папка Killah создана успешно: \(killahDocumentsURL.path)")
                
                // Проверяем, что папка действительно создана
                let created = fileManager.fileExists(atPath: killahDocumentsURL.path)
                print("🔍 Проверка создания папки: \(created)")
                
                // Создаем README файл
                print("📝 Создаем README файл...")
                let readmeContent = """
                # Killah Documents
                
                This folder contains your Killah text editor documents.
                
                Created by Killah Text Editor
                """
                
                let readmePath = killahDocumentsURL.appendingPathComponent("README.md")
                print("📁 Путь к README: \(readmePath.path)")
                
                try readmeContent.write(to: readmePath, atomically: true, encoding: .utf8)
                print("✅ README файл создан при инициализации: \(readmePath.path)")
            } catch {
                print("❌ Ошибка создания папки при инициализации: \(error)")
                print("❌ Детали ошибки: \(error.localizedDescription)")
            }
        } else {
            print("📁 Папка Killah уже существует при инициализации: \(killahDocumentsURL.path)")
        }
    }
}

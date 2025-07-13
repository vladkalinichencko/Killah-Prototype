import SwiftUI
import UniformTypeIdentifiers
import AppKit
import Foundation


struct TextDocument: FileDocument {
    static var readableContentTypes: [UTType] { [.plainText, .rtf] }
    var text: String

    init(text: String = "") {
        let initialText = NSAttributedString(string: text, attributes: [.font: FontManager.shared.defaultEditorFont()])
        self.text = initialText.string
    }
    
    init(fromFileAt url: URL) throws {
        let data = try Data(contentsOf: url)

        if url.pathExtension.lowercased() == "rtf" {
            if let attr = NSAttributedString(rtf: data, documentAttributes: nil) {
                self.text = attr.string
            } else {
                self.text = ""
            }
        } else {
            self.text = String(data: data, encoding: .utf8) ?? ""
        }
    }

    init(configuration: ReadConfiguration) throws {
        if configuration.contentType == .rtf {
            guard let data = configuration.file.regularFileContents,
                  let attr = NSAttributedString(rtf: data, documentAttributes: nil) else {
                text = ""
                return
            }
            text = attr.string
        } else {
            if let data = configuration.file.regularFileContents,
               let str = String(data: data, encoding: .utf8) {
                text = str
            } else {
                text = ""
            }
        }
    }

    func fileWrapper(configuration: WriteConfiguration) throws -> FileWrapper {
        if configuration.contentType == .rtf {
            let attr = NSAttributedString(string: text)
            guard let data = attr.rtf(from: NSRange(location: 0, length: attr.length), documentAttributes: [:]) else {
                throw CocoaError(.fileWriteUnknown)
            }
            return FileWrapper(regularFileWithContents: data)
        } else {
            let data = text.data(using: .utf8) ?? Data()
            return FileWrapper(regularFileWithContents: data)
        }
    }
}


struct DocumentItem: Identifiable {
    let id = UUID()
    let url: URL
    let filename: String
    let contentPreview: String
    let date: Date

    var formattedDate: String {
        let formatter = DateFormatter()
        formatter.dateStyle = .medium
        return formatter.string(from: date)
    }
    
    static func loadFromDirectory() -> [DocumentItem] {
        print("🚀 loadFromDirectory() вызвана")
        
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
        
        // Создаем папку Killah если её нет
        if !folderExists {
            print("📁 Папка Killah не найдена, создаем...")
            do {
                try fileManager.createDirectory(at: killahDocumentsURL, withIntermediateDirectories: true)
                print("✅ Папка Killah создана успешно: \(killahDocumentsURL.path)")
                
                // Проверяем, что папка действительно создана
                let created = fileManager.fileExists(atPath: killahDocumentsURL.path)
                print("🔍 Проверка создания папки: \(created)")
                
                // Создаем файл с логотипом для красивого отображения в Finder
                createFolderIcon(for: killahDocumentsURL)
            } catch {
                print("❌ Ошибка создания папки Killah: \(error)")
                print("❌ Детали ошибки: \(error.localizedDescription)")
                return []
            }
        } else {
            print("📁 Папка Killah уже существует: \(killahDocumentsURL.path)")
        }

        print("📂 Сканируем папку: \(killahDocumentsURL.path)")

        let urls: [URL]
        do {
            urls = try fileManager.contentsOfDirectory(at: killahDocumentsURL, includingPropertiesForKeys: nil)
        } catch {
            print("❌ Ошибка чтения содержимого папки: \(error)")
            print("❌ Детали ошибки: \(error.localizedDescription)")
            return []
        }

        print("🔎 Найдено файлов: \(urls.count)")
        urls.forEach { print(" - \($0.lastPathComponent)") }

        let documents = urls.compactMap { url -> DocumentItem? in
            guard ["txt", "rtf"].contains(url.pathExtension.lowercased()) else { return nil }

            let content = (try? String(contentsOf: url, encoding: .utf8)) ?? "..."
            let preview = content.components(separatedBy: .whitespacesAndNewlines)
                                 .prefix(20)
                                 .joined(separator: " ")

            let date = (try? fileManager.attributesOfItem(atPath: url.path)[.modificationDate] as? Date) ?? Date()

            return DocumentItem(
                url: url,
                filename: url.lastPathComponent,
                contentPreview: preview,
                date: date
            )
        }

        print("📄 Возвращаем \(documents.count) документов")
        return documents
    }
    
    private static func createFolderIcon(for folderURL: URL) {
        print("🎨 createFolderIcon() вызвана для папки: \(folderURL.path)")
        
        // Копируем логотип приложения в папку для иконки
        print("🔍 Ищем иконку приложения в бандле...")
        if let appIconPath = Bundle.main.path(forResource: "app-icon-512", ofType: "png", inDirectory: "Assets.xcassets/AppIcon.appiconset") {
            print("✅ Найдена иконка приложения: \(appIconPath)")
            let iconPath = folderURL.appendingPathComponent("folder-icon.png")
            print("📁 Копируем иконку в: \(iconPath.path)")
            
            do {
                try FileManager.default.copyItem(atPath: appIconPath, toPath: iconPath.path)
                print("✅ Иконка папки создана: \(iconPath.path)")
            } catch {
                print("❌ Ошибка копирования иконки: \(error)")
                print("❌ Детали ошибки: \(error.localizedDescription)")
            }
        } else {
            print("⚠️ Иконка приложения не найдена в бандле")
            print("🔍 Проверяем содержимое бандла...")
            if let bundlePath = Bundle.main.resourcePath {
                print("📂 Путь к ресурсам: \(bundlePath)")
                do {
                    let contents = try FileManager.default.contentsOfDirectory(atPath: bundlePath)
                    print("📁 Содержимое бандла: \(contents)")
                } catch {
                    print("❌ Ошибка чтения содержимого бандла: \(error)")
                }
            }
        }
        
        // Создаем README файл с описанием папки
        print("📝 Создаем README файл...")
        let readmeContent = """
        # Killah Documents
        
        This folder contains your Killah text editor documents.
        
        Created by Killah Text Editor
        """
        
        let readmePath = folderURL.appendingPathComponent("README.md")
        print("📁 Путь к README: \(readmePath.path)")
        
        do {
            try readmeContent.write(to: readmePath, atomically: true, encoding: .utf8)
            print("✅ README файл создан: \(readmePath.path)")
        } catch {
            print("❌ Ошибка создания README файла: \(error)")
            print("❌ Детали ошибки: \(error.localizedDescription)")
        }
    }
}



struct DocumentOpenerView: View {
    var fileURL: URL
    @State private var text: String

    init(fileURL: URL) {
        self.fileURL = fileURL
        if let data = try? Data(contentsOf: fileURL),
           let str = String(data: data, encoding: .utf8) {
            _text = State(initialValue: str)
        } else {
            _text = State(initialValue: "[Ошибка загрузки файла]")
        }
    }

    var body: some View {
        TextEditor(text: $text)
            .padding()
    }
}


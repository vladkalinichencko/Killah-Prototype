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
    let isPersonalized: Bool

    var formattedDate: String {
        let formatter = DateFormatter()
        formatter.dateStyle = .medium
        return formatter.string(from: date)
    }
    
    static func loadFromDirectory() -> [DocumentItem] {
        let fileManager = FileManager.default
        
        // Используем системную папку Documents пользователя
        let documentsURL = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask).first!
        let killahDocumentsURL = documentsURL.appendingPathComponent("Killah")
        
        let folderExists = fileManager.fileExists(atPath: killahDocumentsURL.path)
        
        if !folderExists {
            return []
        }
        
        do {
            let fileURLs = try fileManager.contentsOfDirectory(at: killahDocumentsURL, includingPropertiesForKeys: [.creationDateKey], options: [])
            
            let documents = fileURLs
                .filter { $0.pathExtension == "txt" || $0.pathExtension == "rtf" }
                .sorted { url1, url2 in
                    let date1 = (try? url1.resourceValues(forKeys: [.creationDateKey]).creationDate) ?? Date.distantPast
                    let date2 = (try? url2.resourceValues(forKeys: [.creationDateKey]).creationDate) ?? Date.distantPast
                    return date1 > date2
                }
                .map { url in
                    let content = (try? String(contentsOf: url, encoding: .utf8)) ?? "..."
                    let preview = content.components(separatedBy: .whitespacesAndNewlines)
                        .prefix(20)
                        .joined(separator: " ")
                    let date = (try? fileManager.attributesOfItem(atPath: url.path)[.modificationDate] as? Date) ?? Date()
                    
                    return DocumentItem(
                        url: url,
                        filename: url.lastPathComponent,
                        contentPreview: preview,
                        date: date,
                        isPersonalized: Bool.random() // Временно случайное значение для демонстрации
                    )
                }
            
            return documents
        } catch {
            return []
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


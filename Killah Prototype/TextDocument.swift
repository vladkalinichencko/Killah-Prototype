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
        let fileManager = FileManager.default
        let documentsURL = fileManager.urls(for: .documentDirectory, in: .userDomainMask).first!

        print("📂 Сканируем папку: \(documentsURL.path)")

        let urls = (try? fileManager.contentsOfDirectory(at: documentsURL, includingPropertiesForKeys: nil)) ?? []

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

        return documents
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


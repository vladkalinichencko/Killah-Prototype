import SwiftUI
import UniformTypeIdentifiers
import AppKit

struct TextDocument: FileDocument {
    static var readableContentTypes: [UTType] { [.plainText, .rtf] }
    var text: String

    init(text: String = "") {
        let initialText = NSAttributedString(string: text, attributes: [.font: FontManager.shared.defaultEditorFont()])
        self.text = initialText.string
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

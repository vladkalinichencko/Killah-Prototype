import Foundation
import Combine

class LLMTextTransformer: ObservableObject {
    private var process: Process?
    private var stdinPipe: Pipe?
    private var stdoutPipe: Pipe?

    // Callback: (originalParagraph, transformedParagraph)
    var onParagraphTransformed: ((String, String) -> Void)?

    init() {
        setupPythonProcess()
    }

    private func setupPythonProcess() {
        // Placeholder for Python process management
        print("LLMTextTransformer: Python process setup (simulated).")
    }

    func transformParagraph(_ paragraph: String, userPrompt: String = "Process this text.") {
        // Simulate sending paragraph and prompt to Python LLM script
        print("LLMTextTransformer: Simulating transformation for paragraph: \\(paragraph.prefix(50))...")

        // Simulate a delay and a transformed text
        DispatchQueue.global().asyncAfter(deadline: .now() + 1.0) { [weak self] in
            // Simple simulation: append "[TRANSFORMED]" or make a minor change
            var transformed = paragraph
            if paragraph.lowercased().contains("hello") {
                transformed = paragraph.replacingOccurrences(of: "hello", with: "Greetings", options: .caseInsensitive) + " [LLM Edit]"
            } else if paragraph.count > 10 {
                 // Simulate a "cross-out" and "add" by returning original and modified
                 // For actual diff display, the View needs to compare.
                 // Here, we just provide a modified version.
                 let words = paragraph.split(separator: " ").map(String.init)
                 if words.count > 3 {
                    // "Remove" a word and "add" another
                    // This is a very crude simulation of a diff.
                    // The actual diffing for display (strikethrough, bold) happens in the View
                    // by comparing original and transformed.
                    transformed = words.dropLast().joined(separator: " ") + " ... plus some LLM magic."
                 } else {
                    transformed = paragraph + " (transformed by LLM)."
                 }
            } else {
                 transformed = paragraph + " (transformed by LLM)."
            }
            
            self?.onParagraphTransformed?(paragraph, transformed)
        }
    }

    func stopService() {
        process?.terminate()
        process = nil
        stdinPipe = nil
        stdoutPipe = nil
        print("LLMTextTransformer: Stopped (simulated).")
    }
    
    deinit {
        stopService()
    }
}

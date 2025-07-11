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
        showWelcomeWindow()
    }

    func showWelcomeWindow() {
        guard let deps = Self.dependencies else {
            print("❌ AppDelegate.dependencies не установлены")
            return
        }

        let window = NSWindow(
            contentRect: NSRect(x: 0, y: 0, width: 700, height: 500),
            styleMask: [.titled, .closable, .miniaturizable, .resizable],
            backing: .buffered,
            defer: false
        )
        window.title = "Welcome"
        window.isReleasedWhenClosed = false
        window.center()

        let rootView = WelcomeView(
            onCreateNewFile: {
                NSDocumentController.shared.newDocument(nil)
            },
            onOpenFile: { url in
                NSDocumentController.shared.openDocument(withContentsOf: url, display: true) { _, _, _ in }
            }
        )
        .environmentObject(deps.llmEngine)
        .environmentObject(deps.audioEngine)
        .environmentObject(deps.themeManager)
        .environmentObject(deps.modelManager)

        window.contentView = NSHostingView(rootView: rootView)
        
        window.isOpaque = true
        window.backgroundColor = NSColor.windowBackgroundColor
        window.titlebarAppearsTransparent = false
        window.hasShadow = true
        window.styleMask.remove(.fullSizeContentView)
        
        window.makeKeyAndOrderFront(nil)
    }

}

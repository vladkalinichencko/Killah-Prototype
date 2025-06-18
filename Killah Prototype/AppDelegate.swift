import Cocoa
import SwiftUI

class AppDelegate: NSObject, NSApplicationDelegate {
    var windowController: CustomWindowController?
    var document = TextDocument()

    func applicationDidFinishLaunching(_ notification: Notification) {
        let contentView = NSHostingView(rootView: 
            ContentView(document: .constant(document))
        )
        windowController = CustomWindowController(rootView: contentView)
        windowController?.showWindow(self)
    }
}

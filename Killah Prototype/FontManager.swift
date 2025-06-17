import SwiftUI
import Combine

/// Centralized font size management for the application
class FontManager: ObservableObject {
    static let shared = FontManager()
    
    // Default font sizes
    @Published var defaultEditorFontSize: CGFloat = 20
    @Published var toolbarIconSize: CGFloat = 16
    @Published var promptFieldFontSize: CGFloat = 20
    @Published var menuItemSize: CGFloat = 30
    @Published var promptFieldHeight: CGFloat = 30
    
    // Font size limits
    let minFontSize: CGFloat = 8
    let maxFontSize: CGFloat = 72
    let fontSizeStep: CGFloat = 2
    
    private init() {}
    
    // Helper methods
    func increaseFontSize() {
        defaultEditorFontSize = min(defaultEditorFontSize + fontSizeStep, maxFontSize)
    }
    
    func decreaseFontSize() {
        defaultEditorFontSize = max(defaultEditorFontSize - fontSizeStep, minFontSize)
    }
    
    func systemFont(ofSize size: CGFloat) -> NSFont {
        return NSFont.systemFont(ofSize: size)
    }
    
    func defaultEditorFont() -> NSFont {
        return NSFont.systemFont(ofSize: defaultEditorFontSize)
    }
}

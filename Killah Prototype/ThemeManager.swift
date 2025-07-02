import SwiftUI
import Combine

class ThemeManager: ObservableObject {
    enum Theme: String, CaseIterable, Identifiable {
        case system = "System"
        case light = "Light"
        case dark = "Dark"
        var id: String { self.rawValue }
    }
    
    @Published var currentTheme: Theme {
        didSet {
            // Save the new theme to UserDefaults
            UserDefaults.standard.set(currentTheme.rawValue, forKey: "appTheme")
        }
    }

    init() {
        // Load the saved theme or default to system
        let savedTheme = UserDefaults.standard.string(forKey: "appTheme") ?? "System"
        self.currentTheme = Theme(rawValue: savedTheme) ?? .system
    }

    func applyTheme(to window: NSWindow?) {
        switch currentTheme {
        case .light:
            window?.appearance = NSAppearance(named: .aqua)
        case .dark:
            window?.appearance = NSAppearance(named: .darkAqua)
        case .system:
            window?.appearance = nil // Resets to system appearance
        }
    }
} 
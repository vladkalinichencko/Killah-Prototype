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
            // Apply theme to all windows
            applyThemeToAllWindows()
        }
    }

    init() {
        // Load the saved theme or default to system
        let savedTheme = UserDefaults.standard.string(forKey: "appTheme") ?? "System"
        self.currentTheme = Theme(rawValue: savedTheme) ?? .system
    }

    func applyTheme(to window: NSWindow?) {
        #if os(macOS)
        guard let window = window else { return }
        
        switch currentTheme {
        case .light:
            window.appearance = NSAppearance(named: .aqua)
        case .dark:
            window.appearance = NSAppearance(named: .darkAqua)
        case .system:
            window.appearance = nil // Resets to system appearance
        }
        
        // Force window to update its appearance
        window.invalidateShadow()
        window.display()
        #endif
    }
    
    private func applyThemeToAllWindows() {
        #if os(macOS)
        DispatchQueue.main.async {
            for window in NSApplication.shared.windows {
                self.applyTheme(to: window)
            }
            
            // Также применяем тему к окну настроек, если оно открыто
            if let settingsWindow = NSApplication.shared.windows.first(where: { $0.title == "Settings" }) {
                self.applyTheme(to: settingsWindow)
            }
        }
        #endif
    }
    
    // iOS/iPadOS: возвращает ColorScheme для SwiftUI
    var colorScheme: ColorScheme? {
        switch currentTheme {
        case .light:
            return .light
        case .dark:
            return .dark
        case .system:
            return nil // Использует системную тему
        }
    }
} 
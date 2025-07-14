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
            UserDefaults.standard.set(currentTheme.rawValue, forKey: "appTheme")
            UserDefaults.standard.synchronize()
            applyAppTheme()
        }
    }

    init() {
        let savedThemeRawValue = UserDefaults.standard.string(forKey: "appTheme") ?? Theme.system.rawValue
        self.currentTheme = Theme(rawValue: savedThemeRawValue) ?? .system
        applyAppTheme()
    }

    func applyAppTheme() {
        #if os(macOS)
        let appearance: NSAppearance?
        switch currentTheme {
        case .light:
            appearance = NSAppearance(named: .aqua)
        case .dark:
            appearance = NSAppearance(named: .darkAqua)
        case .system:
            appearance = nil
        }
        NSApplication.shared.appearance = appearance
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
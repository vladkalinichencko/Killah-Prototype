import SwiftUI

struct SettingsView: View {
    @EnvironmentObject var themeManager: ThemeManager
    @ObservedObject var modelManager: ModelManager
    
    var body: some View {
        TabView {
            Form {
                Picker("Theme:", selection: $themeManager.currentTheme) {
                    ForEach(ThemeManager.Theme.allCases) { theme in
                        Text(theme.rawValue).tag(theme)
                    }
                }
                .pickerStyle(SegmentedPickerStyle())
            }
            .padding()
            .tabItem {
                Label("Appearance", systemImage: "paintbrush")
            }

            VStack {
                Text("Model Management")
                    .font(.headline)
                
                // Add content related to model management here
                // For example, show current status and a re-download button.
                Button("Re-check Models") {
                    modelManager.verifyModels()
                }
            }
            .padding()
            .tabItem {
                Label("Models", systemImage: "cpu")
            }
        }
        .frame(width: 400, height: 200)
    }
} 
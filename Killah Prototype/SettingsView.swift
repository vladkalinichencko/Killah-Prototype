import SwiftUI

struct SettingsView: View {
    @EnvironmentObject var themeManager: ThemeManager
    @ObservedObject var modelManager: ModelManager
    @State private var showModelSheet = false
    
    var body: some View {
        Form {
            // Appearance section
            Section(header: Text("Appearance")) {
                Picker("Theme", selection: $themeManager.currentTheme) {
                    ForEach(ThemeManager.Theme.allCases) { theme in
                        Text(theme.rawValue).tag(theme)
                    }
                }
                .pickerStyle(.segmented)
            }

            // Models section
            Section(header: Text("Models")) {
                Group {
                    switch modelManager.status {
                    case .ready:
                        Label("Models are up to date", systemImage: "checkmark.circle.fill")
                            .foregroundColor(.green)
                    case .needsDownloading:
                        Label("Models need to be downloaded", systemImage: "arrow.down.circle.fill")
                            .foregroundColor(.orange)
                        Button("Download Now") { showModelSheet = true }
                    case .downloading(let progress):
                        ProgressView(value: progress) {
                            Text("Downloading…")
                        }
                    case .error(let message):
                        Label(message, systemImage: "xmark.octagon.fill")
                            .foregroundColor(.red)
                        Button("Retry") { modelManager.verifyModels() }
                    case .checking:
                        ProgressView("Checking models…")
                    }
                }

                Button("Force Re-check") {
                    modelManager.verifyModels()
                }
                .controlSize(.small)
            }
        }
        .formStyle(.grouped)
        .frame(width: 420)
        .padding()
        .sheet(isPresented: $showModelSheet) {
            ModelDownloadView(
                modelManager: modelManager,
                missingFiles: (modelManager.status.missingFiles ?? []),
                isDownloading: modelManager.status.isDownloading,
                downloadProgress: modelManager.status.progress
            )
        }
    }
} 
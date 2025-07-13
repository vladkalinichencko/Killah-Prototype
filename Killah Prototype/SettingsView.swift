import SwiftUI

struct SettingsView: View {
    @EnvironmentObject var themeManager: ThemeManager
    @ObservedObject var modelManager: ModelManager
    @State private var showModelSheet = false
    @State private var documentsPath: String = ""
    
    var body: some View {
        Form {
            // Appearance section
            Section(header: Text("Appearance".localized)) {
                Picker("Theme", selection: $themeManager.currentTheme) {
                    ForEach(ThemeManager.Theme.allCases) { theme in
                        Text(theme.rawValue).tag(theme)
                    }
                }
                .pickerStyle(.segmented)
                .onChange(of: themeManager.currentTheme) { _ in
                    applyThemeToAllWindows()
                }
            }

            // Documents section
            Section(header: Text("Documents".localized)) {
                HStack {
                    Text("Default documents folder:".localized)
                    Spacer()
                    Text(documentsPath.isEmpty ? "~/Documents/Killah" : documentsPath)
                        .foregroundColor(.secondary)
                }
                
                HStack {
                    Button("Change folder...".localized) {
                        #if os(macOS)
                        let panel = NSOpenPanel()
                        panel.canChooseFiles = false
                        panel.canChooseDirectories = true
                        panel.allowsMultipleSelection = false
                        panel.message = "Select default documents folder".localized
                        
                        if panel.runModal() == .OK, let url = panel.url {
                            documentsPath = url.path
                            UserDefaults.standard.set(url.path, forKey: "DefaultOpenDirectory")
                            UserDefaults.standard.synchronize()
                        }
                        #else
                        // iOS/iPadOS: используем DocumentPicker
                        // Здесь можно добавить DocumentPicker для iOS
                        print("Document picker for iOS/iPadOS")
                        #endif
                    }
                    
                    Button("Reset to default".localized) {
                        resetDocumentsPath()
                    }
                    .buttonStyle(.bordered)
                }
            }

            // Models section
            Section(header: Text("Models".localized)) {
                Group {
                    switch modelManager.status {
                    case .ready:
                        VStack(alignment: .leading, spacing: 12) {
                            Label("All models are downloaded".localized, systemImage: "checkmark.circle.fill")
                                .foregroundColor(.green)
                            
                            HStack {
                                Button("Delete all models".localized) {
                                    deleteAllModels()
                                }
                                .buttonStyle(.borderedProminent)
                                .tint(.red)
                                
                                Button("Force re-check".localized) {
                                    modelManager.verifyModels()
                                }
                                .buttonStyle(.bordered)
                            }
                        }
                        
                    case .needsDownloading(let missing):
                        VStack(alignment: .leading, spacing: 12) {
                            Label(String(format: "%d models need to be downloaded".localized, missing.count), systemImage: "arrow.down.circle.fill")
                                .foregroundColor(.orange)
                            
                            HStack {
                                Button("Download models".localized) {
                                    modelManager.downloadModels()
                                }
                                .buttonStyle(.borderedProminent)
                                
                                Button("Force re-check".localized) {
                                    modelManager.verifyModels()
                                }
                                .buttonStyle(.bordered)
                            }
                        }
                        
                    case .downloading(let progress):
                        VStack(alignment: .leading, spacing: 12) {
                            ProgressView(value: progress) {
                                Text("Downloading models...".localized)
                            }
                            Text(String(format: "%d%%".localized, Int(progress * 100)))
                                .font(.caption)
                                .foregroundColor(.secondary)
                            
                            Button("Force re-check".localized) {
                                modelManager.verifyModels()
                            }
                            .buttonStyle(.bordered)
                        }
                        
                    case .error(let message):
                        VStack(alignment: .leading, spacing: 12) {
                            Label(message, systemImage: "xmark.octagon.fill")
                                .foregroundColor(.red)
                            
                            HStack {
                                Button("Retry download".localized) {
                                    modelManager.verifyModels()
                                }
                                .buttonStyle(.borderedProminent)
                                
                                Button("Force re-check".localized) {
                                    modelManager.verifyModels()
                                }
                                .buttonStyle(.bordered)
                            }
                        }
                        
                    case .checking:
                        VStack(alignment: .leading, spacing: 12) {
                            ProgressView("Checking models...")
                            
                            Button("Force re-check".localized) {
                                modelManager.verifyModels()
                            }
                            .buttonStyle(.bordered)
                        }
                    }
                }
            }
            
            // Settings info section
            Section(header: Text("Settings Info".localized)) {
                VStack(alignment: .leading, spacing: 8) {
                    HStack {
                        Text("Current theme:".localized)
                        Spacer()
                        Text(themeManager.currentTheme.rawValue)
                            .foregroundColor(.secondary)
                    }
                    
                    HStack {
                        Text("Documents folder:".localized)
                        Spacer()
                        Text(documentsPath.isEmpty ? "Default" : "Custom")
                            .foregroundColor(.secondary)
                    }
                    
                    Button("Reset all settings to default".localized) {
                        resetAllSettings()
                    }
                    .buttonStyle(.bordered)
                    .tint(.orange)
                }
            }
        }
        .formStyle(.grouped)
        .frame(minWidth: 250, minHeight: 450)
        .frame(maxHeight: .infinity)
        .onAppear {
            loadAllSettings()
            applyThemeToAllWindows()
        }
        .onChange(of: themeManager.currentTheme) { _ in
            applyThemeToAllWindows()
        }
        .sheet(isPresented: $showModelSheet) {
            ModelDownloadView(
                modelManager: modelManager,
                missingFiles: modelManager.status.missingFiles ?? [],
                isDownloading: modelManager.status.isDownloading,
                downloadProgress: modelManager.status.progress
            )
        }
    }
    
    private func loadAllSettings() {
        loadDocumentsPath()
    }
    
    private func loadDocumentsPath() {
        documentsPath = UserDefaults.standard.string(forKey: "DefaultOpenDirectory") ?? ""
    }
    
    private func deleteAllModels() {
        #if os(macOS)
        let alert = NSAlert()
        alert.messageText = "Delete All Models".localized
        alert.informativeText = "This will delete all downloaded models. You'll need to download them again to use the app.".localized
        alert.alertStyle = .warning
        alert.addButton(withTitle: "Delete")
        alert.addButton(withTitle: "Cancel")
        
        if alert.runModal() == .alertFirstButtonReturn {
            modelManager.deleteAllModels()
        }
        #else
        // iOS/iPadOS: используем SwiftUI Alert
        // Здесь можно добавить @State для показа SwiftUI Alert
        print("Delete models for iOS/iPadOS")
        #endif
    }
    
    private func applyThemeToAllWindows() {
        DispatchQueue.main.async {
            for window in NSApplication.shared.windows {
                themeManager.applyTheme(to: window)
            }
        }
    }
    
    private func resetDocumentsPath() {
        let defaultPath = FileManager.default.homeDirectoryForCurrentUser.appendingPathComponent("Documents/Killah").path
        documentsPath = defaultPath
        UserDefaults.standard.set(defaultPath, forKey: "DefaultOpenDirectory")
        UserDefaults.standard.synchronize()
    }
    
    private func resetAllSettings() {
        #if os(macOS)
        let alert = NSAlert()
        alert.messageText = "Reset All Settings".localized
        alert.informativeText = "This will reset all application settings to their default values. This cannot be undone.".localized
        alert.alertStyle = .warning
        alert.addButton(withTitle: "Reset")
        alert.addButton(withTitle: "Cancel")
        
        if alert.runModal() == .alertFirstButtonReturn {
            UserDefaults.standard.removePersistentDomain(forName: Bundle.main.bundleIdentifier!)
            UserDefaults.standard.synchronize()
            // Reload settings to reflect changes
            loadAllSettings()
            applyThemeToAllWindows()
            
            // Show success message
            let successAlert = NSAlert()
            successAlert.messageText = "Settings Reset".localized
            successAlert.informativeText = "All settings have been reset to default values. Some changes may require restarting the application.".localized
            successAlert.alertStyle = .informational
            successAlert.addButton(withTitle: "OK")
            successAlert.runModal()
        }
        #else
        // iOS/iPadOS: используем SwiftUI Alert
        print("Reset settings for iOS/iPadOS")
        #endif
    }
} 
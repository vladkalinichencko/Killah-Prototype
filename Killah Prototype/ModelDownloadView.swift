import SwiftUI

struct ModelDownloadView: View {
    @ObservedObject var modelManager: ModelManager
    @EnvironmentObject var appState: AppStateManager
    
    let missingFiles: [ModelManager.ModelFile]
    let isDownloading: Bool
    let downloadProgress: Double
    
    var body: some View {
        VStack(spacing: 20) {
            Text("AI Models Required".localized)
                .font(.title2)
                .fontWeight(.bold)

            Text("To enable AI features like autocompletion and voice commands, additional models need to be downloaded.".localized)
                .multilineTextAlignment(.center)
                .foregroundColor(.secondary)
                .frame(maxWidth: 300)

            if isDownloading {
                ProgressView("Downloading...".localized, value: downloadProgress, total: 1.0)
                    .progressViewStyle(LinearProgressViewStyle())
                    .frame(width: 250)
            } else {
                HStack {
                    Button("Not Now".localized) {
                        appState.closeModelDownloadSheet()
                    }
                    .keyboardShortcut(.cancelAction)

                    Button("Download".localized) {
                        modelManager.downloadModels()
                    }
                    .buttonStyle(.borderedProminent)
                    .keyboardShortcut(.defaultAction)
                }
            }
        }
        .padding(30)
        .frame(width: 400, height: 250)
        .onChange(of: modelManager.status) {
            if modelManager.status == .ready {
                appState.closeModelDownloadSheet()
            }
        }
    }
}

extension ModelManager.ModelFile: Identifiable {
    var id: String { dirName }
}

struct ModelDownloadView_Previews: PreviewProvider {
    static var previews: some View {
        // Create a mock ModelManager for the preview
        let manager = ModelManager()
        
        ModelDownloadView(
            modelManager: manager,
            missingFiles: [],
            isDownloading: false,
            downloadProgress: 0.0
        )
    }
} 
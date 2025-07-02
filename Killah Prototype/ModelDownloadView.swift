import SwiftUI

struct ModelDownloadView: View {
    @Environment(\.dismiss) var dismiss
    @ObservedObject var modelManager: ModelManager
    
    let missingFiles: [ModelManager.ModelFile]
    let isDownloading: Bool
    let downloadProgress: Double
    
    var body: some View {
        VStack(spacing: 20) {
            Text("AI Models Required")
                .font(.title2)
                .fontWeight(.bold)

            Text("To enable AI features like autocompletion and voice commands, additional models need to be downloaded.")
                .multilineTextAlignment(.center)
                .foregroundColor(.secondary)
                .frame(maxWidth: 300)

            if isDownloading {
                ProgressView("Downloading...", value: downloadProgress, total: 1.0)
                    .progressViewStyle(LinearProgressViewStyle())
                    .frame(width: 250)
            } else {
                HStack {
                    Button("Not Now") {
                        dismiss()
                    }
                    .keyboardShortcut(.cancelAction)

                    Button("Download") {
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
                dismiss()
            }
        }
    }
}

extension ModelManager.ModelFile: Identifiable {
    var id: String { name }
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
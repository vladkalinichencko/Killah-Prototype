import SwiftUI

struct ModelDownloadView: View {
    @ObservedObject var modelManager: ModelManager
    
    let missingFiles: [ModelManager.ModelFile]
    let isDownloading: Bool
    let downloadProgress: Double
    
    var body: some View {
        VStack(spacing: 20) {
            Text("Model Download Required")
                .font(.title2)
                .fontWeight(.bold)
            
            if isDownloading {
                ProgressView("Downloading...", value: downloadProgress, total: 1.0)
                    .progressViewStyle(LinearProgressViewStyle())
                    .frame(width: 250)
            } else {
                Text("The following models need to be downloaded to continue:")
                    .multilineTextAlignment(.center)
                    .foregroundColor(.secondary)
                
                List(missingFiles, id: \.name) { file in
                    Text(file.name)
                        .font(.mono)
                }
                .listStyle(.bordered(alternatesRowBackgrounds: true))
                .frame(height: 150)

                Button("Download Models") {
                    modelManager.downloadModels()
                }
                .buttonStyle(.borderedProminent)
            }
        }
        .padding(30)
        .frame(width: 400, height: 300)
    }
}

extension ModelManager.ModelFile: Identifiable {
    var id: String { name }
} 
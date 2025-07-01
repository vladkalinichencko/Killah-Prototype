import Foundation
import Combine
import ZIPFoundation

class ModelManager: ObservableObject {

    enum ModelStatus {
        case checking
        case needsDownloading(missing: [ModelFile])
        case downloading(progress: Double)
        case ready
        case error(String)
    }

    struct ModelFile {
        let name: String
        let remoteURL: URL
        // We can add checksums later for validation
        // let checksum: String
    }

    @Published var status: ModelStatus = .checking
    
    // The list of models to be downloaded from the cloud.
    // Directories like model weights should be zipped for easier download.
    private var allModels: [ModelFile] = [
        ModelFile(name: "wav2vec2-xls-r-300m.zip", remoteURL: URL(string: "https://huggingface.co/facebook/wav2vec2-xls-r-300m/archive/main.zip")!),
        ModelFile(name: "gemma-3-4b-pt-q8bits.zip", remoteURL: URL(string: "https://huggingface.co/poinka/gemma-3-4b-pt-q8bits/archive/main.zip")!)
        // Any other models like MLP projectors or LoRA adapters can be added here.
        // If they are single files, they don't need to be zipped.
    ]

    private var downloadTasks: [URLSessionDownloadTask] = []
    
    private lazy var modelsDirectory: URL = {
        let fileManager = FileManager.default
        let appSupportURL = fileManager.urls(for: .applicationSupportDirectory, in: .userDomainMask).first!
        let directory = appSupportURL.appendingPathComponent("KillahPrototype/models")
        
        // Create the directory if it doesn't exist
        if !fileManager.fileExists(atPath: directory.path) {
            try? fileManager.createDirectory(at: directory, withIntermediateDirectories: true, attributes: nil)
        }
        
        return directory
    }()

    func getModelsDirectory() -> URL {
        return modelsDirectory
    }

    func getModelPath(for modelName: String) -> String? {
        let fileURL = modelsDirectory.appendingPathComponent(modelName)
        return FileManager.default.fileExists(atPath: fileURL.path) ? fileURL.path : nil
    }

    func verifyModels() {
        var missingFiles: [ModelFile] = []
        for model in allModels {
            let localPath = getModelPath(for: model.name)
            if localPath == nil {
                // Special handling for zipped directories
                if model.name.hasSuffix(".zip") {
                    let dirName = model.name.replacingOccurrences(of: ".zip", with: "")
                    if getModelPath(for: dirName) == nil {
                        missingFiles.append(model)
                    }
                } else {
                    missingFiles.append(model)
                }
            }
        }

        if missingFiles.isEmpty {
            DispatchQueue.main.async { self.status = .ready }
        } else {
            DispatchQueue.main.async { self.status = .needsDownloading(missing: missingFiles) }
        }
    }

    func downloadModels() {
        guard case .needsDownloading(let missing) = status else { return }
        
        DispatchQueue.main.async {
            self.status = .downloading(progress: 0)
        }
        
        // This is a simplified download logic. For a real app, you'd want a more robust
        // solution that handles individual progress, errors, and decompression.
        let session = URLSession(configuration: .default, delegate: self, delegateQueue: nil)
        downloadTasks = missing.map { session.downloadTask(with: $0.remoteURL) }
        downloadTasks.forEach { $0.resume() }
    }
}

extension ModelManager: URLSessionDownloadDelegate {
    func urlSession(_ session: URLSession, downloadTask: URLSessionDownloadTask, didFinishDownloadingTo location: URL) {
        guard let modelName = downloadTask.originalRequest?.url?.lastPathComponent else { return }

        let destinationURL = modelsDirectory.appendingPathComponent(modelName)
        let fileManager = FileManager.default
        
        do {
            try? fileManager.removeItem(at: destinationURL)
            try fileManager.moveItem(at: location, to: destinationURL)
            
            if destinationURL.pathExtension == "zip" {
                try unzip(file: destinationURL)
                try fileManager.removeItem(at: destinationURL) // Clean up the zip file
            }
        } catch {
            DispatchQueue.main.async {
                self.status = .error("Failed to move or unzip file: \(error.localizedDescription)")
            }
            return
        }

        // Simple check: assume if one download finishes, we check all.
        // A more robust implementation would wait for all tasks to complete.
        DispatchQueue.main.asyncAfter(deadline: .now() + 1.0) { // Add a small delay for file system to catch up
            self.verifyModels()
        }
    }

    func urlSession(_ session: URLSession, downloadTask: URLSessionDownloadTask, didWriteData bytesWritten: Int64, totalBytesWritten: Int64, totalBytesExpectedToWrite: Int64) {
        // This provides progress for a single download. To get total progress,
        // we'd need to track the progress of all downloads.
        let progress = Double(totalBytesWritten) / Double(totalBytesExpectedToWrite)
        
        DispatchQueue.main.async {
            // For simplicity, showing progress of the first download.
            if self.downloadTasks.first == downloadTask {
                self.status = .downloading(progress: progress)
            }
        }
    }
    
    func urlSession(_ session: URLSession, task: URLSessionTask, didCompleteWithError error: Error?) {
        if let error = error {
            DispatchQueue.main.async {
                self.status = .error(error.localizedDescription)
            }
        }
    }
    
    private func unzip(file: URL) throws {
        let fileManager = FileManager()
        let destinationURL = file.deletingPathExtension()
        try fileManager.createDirectory(at: destinationURL, withIntermediateDirectories: true, attributes: nil)
        try fileManager.unzipItem(at: file, to: destinationURL)
        
        // After unzipping from Hugging Face, we often get a single sub-folder with a long name like 'repo-main'.
        // Let's move its contents up to our destination directory for cleaner paths.
        let unzippedContents = try fileManager.contentsOfDirectory(at: destinationURL, includingPropertiesForKeys: [.isDirectoryKey], options: .skipsHiddenFiles)
        if let repoFolder = unzippedContents.first, (try repoFolder.resourceValues(forKeys: [.isDirectoryKey])).isDirectory == true {
            let innerContents = try fileManager.contentsOfDirectory(at: repoFolder, includingPropertiesForKeys: nil, options: .skipsHiddenFiles)
            for item in innerContents {
                let destinationItemURL = destinationURL.appendingPathComponent(item.lastPathComponent)
                try? fileManager.removeItem(at: destinationItemURL) // remove if exists
                try fileManager.moveItem(at: item, to: destinationItemURL)
            }
            try fileManager.removeItem(at: repoFolder)
        }
    }
} 
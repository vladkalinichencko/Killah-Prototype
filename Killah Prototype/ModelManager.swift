import Foundation
import Combine
import ZIPFoundation

class ModelManager: NSObject, ObservableObject {

    enum ModelStatus: Equatable {
        case checking
        case needsDownloading(missing: [ModelFile])
        case downloading(progress: Double)
        case ready
        case error(String)
    }

    struct ModelFile: Equatable {
        let name: String
        let remoteURL: URL
        // We can add checksums later for validation
        // let checksum: String
    }

    @Published var status: ModelStatus = .checking
    
    // The list of models to be downloaded from the cloud.
    // Directories like model weights should be zipped for easier download.
    private var allModels: [ModelFile] = [
        ModelFile(name: "wav2vec2-xls-r-300m.zip", remoteURL: URL(string: "https://huggingface.co/facebook/wav2vec2-xls-r-300m/resolve/main/pytorch_model.bin")!),
        ModelFile(name: "gemma-3-4b-pt-q8bits.zip", remoteURL: URL(string: "https://huggingface.co/poinka/gemma-3-4b-pt-q8bits/resolve/main/model.safetensors")!)
        // Any other models like MLP projectors or LoRA adapters can be added here.
        // If they are single files, they don't need to be zipped.
    ]

    private var downloadTasks: [URLSessionDownloadTask] = []
    private let downloadGroup = DispatchGroup()
    private var totalBytesToDownload: Int64 = 0
    private var totalBytesDownloaded: Int64 = 0
    private var downloadProgressPerTask: [URLSessionTask: Int64] = [:]
    
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
        DispatchQueue.global(qos: .userInitiated).async {
            var missingFiles: [ModelFile] = []
            for model in self.allModels {
                let localPath = self.getModelPath(for: model.name)
                if localPath == nil {
                    // Special handling for zipped directories
                    if model.name.hasSuffix(".zip") {
                        let dirName = model.name.replacingOccurrences(of: ".zip", with: "")
                        if self.getModelPath(for: dirName) == nil {
                            missingFiles.append(model)
                        }
                    } else {
                        missingFiles.append(model)
                    }
                }
            }
            
            DispatchQueue.main.async {
                if missingFiles.isEmpty {
                    self.status = .ready
                } else {
                    self.status = .needsDownloading(missing: missingFiles)
                }
            }
        }
    }

    func downloadModels() {
        guard case .needsDownloading(let missing) = status else { return }
        
        DispatchQueue.main.async {
            self.status = .downloading(progress: 0)
        }
        
        self.totalBytesDownloaded = 0
        self.downloadProgressPerTask = [:]

        let session = URLSession(configuration: .default, delegate: self, delegateQueue: nil)
        downloadTasks = missing.map { modelFile in
            self.downloadGroup.enter()
            let task = session.downloadTask(with: modelFile.remoteURL)
            return task
        }
        
        // Use an async task to get estimated sizes
        Task {
            self.totalBytesToDownload = await getTotalSize(for: missing)
            downloadTasks.forEach { $0.resume() }
        }

        downloadGroup.notify(queue: .main) {
            // This closure is called when all download tasks are complete
            self.verifyModels()
        }
    }

    private func getTotalSize(for models: [ModelFile]) async -> Int64 {
        var totalSize: Int64 = 0
        for model in models {
            var request = URLRequest(url: model.remoteURL)
            request.httpMethod = "HEAD"
            if let (_, response) = try? await URLSession.shared.data(for: request),
               let httpResponse = response as? HTTPURLResponse,
               let length = httpResponse.value(forHTTPHeaderField: "Content-Length"),
               let bytes = Int64(length) {
                totalSize += bytes
            }
        }
        return totalSize > 0 ? totalSize : 1 // Avoid division by zero
    }
}

extension ModelManager: URLSessionDownloadDelegate {
    func urlSession(_ session: URLSession, downloadTask: URLSessionDownloadTask, didFinishDownloadingTo location: URL) {
        defer {
            downloadGroup.leave()
        }

        guard let url = downloadTask.originalRequest?.url,
              let model = allModels.first(where: { $0.remoteURL == url }) else {
            DispatchQueue.main.async {
                self.status = .error("Could not identify downloaded model for URL: \(downloadTask.originalRequest?.url?.absoluteString ?? "N/A")")
            }
            return
        }

        let destinationURL = modelsDirectory.appendingPathComponent(model.name)
        let fileManager = FileManager.default
        
        do {
            try? fileManager.removeItem(at: destinationURL)
            try fileManager.moveItem(at: location, to: destinationURL)
            
            if destinationURL.pathExtension == "zip" {
                // Adjust unzipping logic if needed, as we are downloading raw files now.
                // try unzip(file: destinationURL)
                // try fileManager.removeItem(at: destinationURL) 
            }
        } catch {
            DispatchQueue.main.async {
                self.status = .error("Failed to move file: \(error.localizedDescription)")
            }
            return
        }
    }

    func urlSession(_ session: URLSession, downloadTask: URLSessionDownloadTask, didWriteData bytesWritten: Int64, totalBytesWritten: Int64, totalBytesExpectedToWrite: Int64) {
        
        downloadProgressPerTask[downloadTask] = totalBytesWritten
        
        let currentTotalDownloaded = downloadProgressPerTask.values.reduce(0, +)
        
        // If we couldn't get the total size via HEAD requests, use the session's expected size.
        // This is less accurate for multiple files but better than nothing.
        let effectiveTotalSize = totalBytesToDownload > 0 ? totalBytesToDownload : totalBytesExpectedToWrite
        
        let progress = Double(currentTotalDownloaded) / Double(effectiveTotalSize)
        
        DispatchQueue.main.async {
            self.status = .downloading(progress: min(progress, 1.0))
        }
    }
    
    func urlSession(_ session: URLSession, task: URLSessionTask, didCompleteWithError error: Error?) {
        if let error = error {
            defer {
                downloadGroup.leave()
            }
            DispatchQueue.main.async {
                print("Download error for \(task.originalRequest?.url?.absoluteString ?? "unknown URL"): \(error.localizedDescription)")
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
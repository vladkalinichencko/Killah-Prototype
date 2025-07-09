import Foundation
import Combine

class ModelManager: NSObject, ObservableObject {

    enum ModelStatus: Equatable {
        case checking
        case needsDownloading(missing: [ModelFile])
        case downloading(progress: Double)
        case ready
        case error(String)
    }

    struct ModelFile: Equatable {
        /// Directory name where all files for this model will be stored locally
        let dirName: String
        /// Repository ID on Hugging Face (e.g. "facebook/wav2vec2-xls-r-300m")
        let repoID: String
        /// REQUIRED files that must exist inside dirName for the model to be considered complete
        let requiredFiles: [String]
        /// Builds a full remote URL for a concrete file inside the repo
        func remoteURL(for fileName: String) -> URL {
            // Use /resolve/main/<file> endpoint so that we download raw bytes, not html
            return URL(string: "https://huggingface.co/\(repoID)/resolve/main/\(fileName)")!
        }
    }

    @Published var status: ModelStatus = .checking
    
    // The list of models to be downloaded from the cloud.
    // Directories like model weights should be zipped for easier download.
    private var allModels: [ModelFile] = [
        ModelFile(
            dirName: "wav2vec2-xls-r-300m",
            repoID: "facebook/wav2vec2-xls-r-300m",
            requiredFiles: [
                "config.json",
                "preprocessor_config.json",
                "pytorch_model.bin"
            ]
        ),
//        ModelFile(
//            dirName: "gemma-3-4b-pt-q8bits",
//            repoID: "poinka/gemma-3-4b-pt-q8bits",
//            requiredFiles: [
//                "config.json",
//                "tokenizer.json",
//                "tokenizer.model",
//                "tokenizer_config.json",
//                "special_tokens_map.json",
//                "added_tokens.json",
//                "generation_config.json",
//                "pytorch_model.bin",
//                "model.safetensors",
//                "model.gguf"
//            ]
//        )
            ModelFile(
                dirName: "gemma-3-4b-pt-qat-q4_0-gguf",
                repoID: "google/gemma-3-4b-pt-qat-q4_0-gguf",
                requiredFiles: ["gemma-3-4b-pt-q4_0.gguf"]
        )
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
        print("📂 ModelManager.verifyModels() called")
        DispatchQueue.global(qos: .userInitiated).async {
            var missingFiles: [ModelFile] = []
            for model in self.allModels {
                print("🔍 Checking model directory: \(model.dirName)")
                if let dirPath = self.getModelPath(for: model.dirName) {
                    if self.requiredFilesPresent(model: model, in: dirPath) {
                        continue
                    } else {
                        print("⚠️ Directory \(model.dirName) exists but missing required files")
                        missingFiles.append(model)
                        continue
                    }
                } else {
                    // каталога нет, нужно скачать
                    missingFiles.append(model)
                }
                continue // переходим к следующей модели
            }
            
            DispatchQueue.main.async {
                print("✅ Verification finished. Missing files: \(missingFiles.map { $0.dirName })")
                if missingFiles.isEmpty {
                    self.status = .ready
                } else {
                    self.status = .needsDownloading(missing: missingFiles)
                }
            }
        }
    }

    func downloadModels() {
        print("⬇️ ModelManager.downloadModels() initiated")
        guard case .needsDownloading(let missing) = status else { return }
        
        print("Files to download: \(missing.map { $0.dirName })")
        
        DispatchQueue.main.async {
            self.status = .downloading(progress: 0)
        }
        
        self.totalBytesDownloaded = 0
        self.downloadProgressPerTask = [:]

        let session = URLSession(configuration: .default, delegate: self, delegateQueue: nil)

        var tasks: [URLSessionDownloadTask] = []
        var modelsToSize: [(ModelFile, String, URL)] = []

        for model in missing {
            let modelDirURL = self.modelsDirectory.appendingPathComponent(model.dirName)
            try? FileManager.default.createDirectory(at: modelDirURL, withIntermediateDirectories: true, attributes: nil)

            for fileName in model.requiredFiles {
                let localPath = modelDirURL.appendingPathComponent(fileName).path
                if FileManager.default.fileExists(atPath: localPath) { continue }
                let remoteURL = model.remoteURL(for: fileName)
                let task = session.downloadTask(with: remoteURL)
                task.taskDescription = "\(model.dirName)|\(fileName)"
                downloadGroup.enter()
                tasks.append(task)
                modelsToSize.append((model, fileName, remoteURL))
            }
        }

        if tasks.isEmpty {
            DispatchQueue.main.async {
                self.status = .ready
            }
            return
        }

        self.downloadTasks = tasks

        Task {
            self.totalBytesToDownload = await getTotalSizeForFiles(modelsToSize.map { $0.2 })
            tasks.forEach { $0.resume() }
        }

        downloadGroup.notify(queue: .main) {
            // This closure is called when all download tasks are complete
            self.verifyModels()
        }
    }

    private func getTotalSizeForFiles(_ urls: [URL]) async -> Int64 {
        var totalSize: Int64 = 0
        for url in urls {
            var request = URLRequest(url: url)
            request.httpMethod = "HEAD"
            if let (_, response) = try? await URLSession.shared.data(for: request),
               let httpResponse = response as? HTTPURLResponse,
               let length = httpResponse.value(forHTTPHeaderField: "Content-Length"),
               let bytes = Int64(length) {
                totalSize += bytes
            }
        }
        return totalSize > 0 ? totalSize : 1
    }

    private func requiredFilesPresent(model: ModelFile, in dirPath: String) -> Bool {
        let fm = FileManager.default
        for req in model.requiredFiles {
            let fullPath = (dirPath as NSString).appendingPathComponent(req)
            if !fm.fileExists(atPath: fullPath) {
                print("⛔️ Missing required file: \(req) in \(dirPath)")
                return false
            }
        }
        return true
    }
}

extension ModelManager: URLSessionDownloadDelegate {
    func urlSession(_ session: URLSession, downloadTask: URLSessionDownloadTask, didFinishDownloadingTo location: URL) {
        defer {
            downloadGroup.leave()
        }

        guard let description = downloadTask.taskDescription,
              let pipeIndex = description.firstIndex(of: "|") else {
            downloadGroup.leave()
            return
        }

        let modelDirName = String(description[..<pipeIndex])
        let fileName = String(description[description.index(after: pipeIndex)...])

        let destinationDir = modelsDirectory.appendingPathComponent(modelDirName)
        let destinationURL = destinationDir.appendingPathComponent(fileName)

        let fileManager = FileManager.default
        
        do {
            try? fileManager.removeItem(at: destinationURL)
            try fileManager.moveItem(at: location, to: destinationURL)
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
        // This function is kept for backward compatibility but is no longer used in the new per-file download flow.
    }
} 

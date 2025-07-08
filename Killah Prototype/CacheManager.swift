// New CacheManager class
import Foundation

class CacheManager {
    private let suggestionCache: NSCache<NSString, NSString>
    static let shared = CacheManager()
    
    private init() {
        suggestionCache = NSCache()
        suggestionCache.countLimit = 1000 // Adjust based on memory needs
    }
    
    func getCachedSuggestion(for prompt: String, temperature: Float) -> String? {
        let key = "\(prompt)_\(temperature)".sha256() as NSString
        return suggestionCache.object(forKey: key) as String?
    }
    
    func setCachedSuggestion(_ suggestion: String, for prompt: String, temperature: Float) {
        let key = "\(prompt)_\(temperature)".sha256() as NSString
        suggestionCache.setObject(suggestion as NSString, forKey: key)
    }
    
    func invalidateCache() {
        suggestionCache.removeAllObjects()
    }
}

// SHA-256 extension for cache key
import CryptoKit
extension String {
    func sha256() -> String {
        let data = Data(self.utf8)
        let hash = SHA256.hash(data: data)
        return hash.compactMap { String(format: "%02x", $0) }.joined()
    }
}

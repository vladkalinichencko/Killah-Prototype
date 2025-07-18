import SwiftData
import Foundation

@Model
class Embedding {
    @Attribute(.unique) var documentID: String
    var embeddingData: Data
    var isPersonalized: Bool
    var documentURL: URL
    
    init(documentID: String, embeddingData: Data, isPersonalized: Bool, documentURL: URL) {
        self.documentID = documentID
        self.embeddingData = embeddingData
        self.isPersonalized = isPersonalized
        self.documentURL = documentURL
    }
}

//
//  WelcomeView.swift
//  Killah Prototype
//
//  Created by Arthur on 7/10/25.
//

import SwiftUI
import AppKit

struct WelcomeView: View {
    @State private var recentDocuments: [DocumentItem] = []
    @State private var showingFileImporter = false
    @EnvironmentObject var appState: AppStateManager
    @EnvironmentObject var llmEngine: LLMEngine
    @EnvironmentObject var audioEngine: AudioEngine
    @EnvironmentObject var themeManager: ThemeManager
    @EnvironmentObject var modelManager: ModelManager

    var body: some View {
        VStack(spacing: 32) {
            Spacer()
            
            // Сетка недавних документов
            if !recentDocuments.isEmpty {
                VStack(alignment: .leading, spacing: 12) {
                    Text("Недавние документы")
                        .font(.headline)
                        .padding(.leading)

                    ScrollView {
                        LazyVGrid(columns: [GridItem(.adaptive(minimum: 160))], spacing: 20) {
                            ForEach(recentDocuments.prefix(8)) { doc in
                                DocumentCard(document: doc)
                                    .onTapGesture {
                                        appState.openDocument(from: doc.url)
                                    }
                            }
                        }
                        .padding(.horizontal)
                    }
                }
            }
            
            Spacer()
            
            // Две основные кнопки внизу
            HStack(spacing: 24) {
                Button {
                    appState.createNewDocument()
                } label: {
                    Label("Создать новый файл", systemImage: "doc.badge.plus")
                }
                .buttonStyle(.borderedProminent)

                Button {
                    showingFileImporter = true
                } label: {
                    Label("Открыть файл", systemImage: "folder")
                }
                .buttonStyle(.bordered)
            }
            .padding(.bottom, 40)
        }
        .frame(minWidth: 800, minHeight: 600)
        .onAppear {
            print("🚀 WelcomeView.onAppear() вызван")
            recentDocuments = DocumentItem.loadFromDirectory()
            print("📄 WelcomeView: Загружено \(recentDocuments.count) документов")
        }
        .fileImporter(
            isPresented: $showingFileImporter,
            allowedContentTypes: [.plainText, .rtf],
            allowsMultipleSelection: false
        ) { result in
            if case .success(let urls) = result, let url = urls.first {
                appState.openDocument(from: url)
            }
        }
    }
}


struct FileSectionView: View {
    let title: String?
    let items: ArraySlice<DocumentItem>
    let onOpen: (URL) -> Void

    var columns: [GridItem] = Array(repeating: GridItem(.flexible(), spacing: 24), count: 4)

    var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            if let title = title {
                Text(title)
                    .font(.headline)
                    .padding(.leading, 8)
            }

            LazyVGrid(columns: columns, spacing: 24) {
                ForEach(items) { doc in
                    DocumentCard(document: doc)
                        .onTapGesture {
                            onOpen(doc.url)
                        }
                }
            }
        }
    }
}

struct DocumentCard: View {
    let document: DocumentItem

    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            RoundedRectangle(cornerRadius: 12)
                .fill(Color.gray.opacity(0.1))
                .frame(height: 140)
                .overlay(
                    Text(document.contentPreview)
                        .font(.caption)
                        .foregroundColor(.gray)
                        .padding(8)
                        .multilineTextAlignment(.leading)
                )

            Text(document.filename)
                .font(.subheadline)
                .bold()

            Text(document.formattedDate)
                .font(.caption)
                .foregroundColor(.gray)
        }
        .frame(width: 160)
    }
}

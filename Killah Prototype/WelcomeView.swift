//
//  WelcomeView.swift
//  Killah Prototype
//
//  Created by Arthur on 7/10/25.
//

import SwiftUI

struct DocumentHostView: View {
    @Binding var document: TextDocument
    @State private var hasBeenOpened = false

    var body: some View {
        if hasBeenOpened {
            AnyView(ContentView(document: $document))
        } else {
            AnyView(
                WelcomeView(
                    onCreateNewFile: {
                        // Например: просто сбрасываем document.text = ""
                        document.text = ""
                        hasBeenOpened = true
                    },
                    onOpenFile: { url in
                        if let text = try? String(contentsOf: url, encoding: .utf8) {
                            document.text = text
                            hasBeenOpened = true
                        }
                    },
                )
            )
        }
    }
}


struct WelcomeView: View {
    var onCreateNewFile: () -> Void
    var onOpenFile: (URL) -> Void

    @State private var recentDocuments: [DocumentItem] = []
    @State private var showingFileImporter = false


    var body: some View {
        VStack(spacing: 32) {
            // Заголовок
            Text("Killah")
                .font(.system(size: 40, weight: .bold))

            // Две основные кнопки
            HStack(spacing: 24) {
                Button {
                    onCreateNewFile()
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

            Divider()
                .padding(.horizontal, 60)

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
                                        onOpenFile(doc.url)
                                    }
                            }
                        }
                        .padding(.horizontal)
                    }
                }
            }

            Spacer()
        }
        .frame(minWidth: 800, minHeight: 600)
        .onAppear {
            recentDocuments = DocumentItem.loadFromDirectory()
        }
        .fileImporter(
            isPresented: $showingFileImporter,
            allowedContentTypes: [.plainText, .rtf],
            allowsMultipleSelection: false
        ) { result in
            if case .success(let urls) = result, let url = urls.first {
                onOpenFile(url)
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

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
        ZStack {
            // Фон, соответствующий титлбару
            Color.clear
                .background(.ultraThinMaterial)
                .ignoresSafeArea()
            
            // Основной контент
            VStack(spacing: 0) {
                ScrollView {
                    LazyVStack(spacing: 0, pinnedViews: [.sectionHeaders]) {
                        ForEach(groupedDocuments.keys.sorted(by: { $0.rawValue < $1.rawValue }), id: \.self) { timeGroup in
                            if let documents = groupedDocuments[timeGroup], !documents.isEmpty {
                                Section {
                                    LazyVGrid(columns: [GridItem(.adaptive(minimum: 160))], spacing: 20) {
                                        ForEach(documents) { doc in
                                            DocumentCard(document: doc)
                                                .onTapGesture {
                                                    appState.openDocument(from: doc.url)
                                                }
                                        }
                                    }
                                    .padding(.horizontal, 20)
                                    .padding(.top, 16)
                                    .padding(.bottom, 30)
                                } header: {
                                    HStack {
                                        Text(timeGroup.displayName)
                                            .font(.headline)
                                            .foregroundColor(.secondary)
                                            .padding(.horizontal, 20)
                                            .padding(.vertical, 12)
                                        
                                        // Горизонтальный разделитель
                                        Rectangle()
                                            .fill(Color.secondary.opacity(0.3))
                                            .frame(height: 1)
                                            .frame(maxWidth: .infinity)
                                            .padding(.trailing, 20)
                                        
                                        Spacer()
                                    }
                                    .frame(maxWidth: .infinity)
                                }
                            }
                        }
                    }
                }
            }
            .frame(maxWidth: .infinity, maxHeight: .infinity)
            
            // Кнопки в правом нижнем углу
            VStack {
                Spacer()
                HStack {
                    Spacer()
                    VStack(spacing: 16) {
                        Button {
                            appState.createNewDocument()
                        } label: {
                            Image(systemName: "plus")
                                .font(.title2)
                                .foregroundColor(.accentColor)
                                .frame(width: 56, height: 56)
                                .background(
                                    Circle()
                                        .fill(.ultraThinMaterial)
                                        .shadow(radius: 8)
                                )
                        }
                        .buttonStyle(PlainButtonStyle())
                        
                        Button {
                            showingFileImporter = true
                        } label: {
                            Image(systemName: "folder")
                                .font(.title2)
                                .foregroundColor(.accentColor)
                                .frame(width: 56, height: 56)
                                .background(
                                    Circle()
                                        .fill(.ultraThinMaterial)
                                        .shadow(radius: 8)
                                )
                        }
                        .buttonStyle(PlainButtonStyle())
                    }
                    .padding(.trailing, 40)
                    .padding(.bottom, 40)
                }
            }
        }
        .frame(minWidth: 800, minHeight: 600)
        
        // Глобальные View для загрузки
        .overlay(
            VStack {
                Spacer()
                
                if appState.isPythonScriptsStarting {
                    LoadingOverlayView()
                        .padding(.bottom, 20)
                }
            }
        )
        .onAppear {
            recentDocuments = DocumentItem.loadFromDirectory()
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
        .sheet(isPresented: Binding(
            get: { appState.isModelDownloading },
            set: { _ in }
        )) {
            ModelDownloadView(
                modelManager: modelManager,
                missingFiles: (modelManager.status.missingFiles ?? []),
                isDownloading: modelManager.status.isDownloading,
                downloadProgress: modelManager.status.progress
            )
            .environmentObject(appState)
        }
    }
    
    private var groupedDocuments: [TimeGroup: [DocumentItem]] {
        let calendar = Calendar.current
        let now = Date()
        
        return Dictionary(grouping: recentDocuments) { document in
            let daysDifference = calendar.dateComponents([.day], from: document.date, to: now).day ?? 0
            
            switch daysDifference {
            case 0:
                return .today
            case 1:
                return .yesterday
            case 2...7:
                return .lastWeek
            case 8...30:
                return .lastMonth
            default:
                return .older
            }
        }
    }
}

enum TimeGroup: Int, CaseIterable {
    case today = 0
    case yesterday = 1
    case lastWeek = 2
    case lastMonth = 3
    case older = 4
    
    var displayName: String {
        switch self {
        case .today:
            return "Сегодня"
        case .yesterday:
            return "Вчера"
        case .lastWeek:
            return "Последние 7 дней"
        case .lastMonth:
            return "Последние 30 дней"
        case .older:
            return "Ранее"
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
                    VStack(alignment: .leading, spacing: 4) {
                        Text(document.contentPreview)
                            .font(.caption)
                            .foregroundColor(.gray)
                            .multilineTextAlignment(.leading)
                            .lineLimit(6)
                        
                        Spacer()
                        
                        HStack {
                            Text(document.formattedDate)
                                .font(.caption2)
                                .foregroundColor(.secondary)
                            Spacer()
                        }
                    }
                    .padding(8)
                )

            VStack(alignment: .leading, spacing: 4) {
                Text(document.filename)
                    .font(.subheadline)
                    .bold()
                    .lineLimit(1)
                
                // Статус персонализации
                HStack(spacing: 4) {
                    Image(systemName: document.isPersonalized ? "checkmark.circle.fill" : "xmark.circle.fill")
                        .font(.caption2)
                        .foregroundColor(.secondary)
                    
                    Text(document.isPersonalized ? "Персонализирован" : "Не персонализирован")
                        .font(.caption2)
                        .foregroundColor(.secondary)
                        .lineLimit(1)
                }
            }
        }
        .frame(width: 160)
        .contentShape(Rectangle())
    }
}

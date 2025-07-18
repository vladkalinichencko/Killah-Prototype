//
//  WelcomeView.swift
//  Killah Prototype
//
//  Created by Arthur on 7/10/25.
//

import SwiftUI
import AppKit
import SwiftData

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
                .background(.regularMaterial)
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
                                    // Material background appears only when header is really pinned
                                    .background(
                                        GeometryReader { proxy in
                                            Rectangle()
                                                .fill(.regularMaterial)
                                                .opacity(proxy.frame(in: .named("scroll")).minY <= 0 ? 1 : 0)
                                        }
                                    )
                                }
                            }
                        }
                    }
                    // Required coordinate space for pinned-header detection
                }
                .coordinateSpace(name: "scroll")
                .padding(.top, 1) // fixed offset below title bar
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
                                        .shadow(color: Color.black.opacity(0.2), radius: 14, x: 0, y: 6)
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
                                        .shadow(color: Color.black.opacity(0.2), radius: 14, x: 0, y: 6)
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
            if calendar.isDateInToday(document.date) {
                return .today
            } else if calendar.isDateInYesterday(document.date) {
                return .yesterday
            } else {
                let daysDifference = calendar.dateComponents([.day], from: document.date, to: now).day ?? 0
                switch daysDifference {
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
    @EnvironmentObject var llmEngine: LLMEngine
    @Environment(\.modelContext) var context
    
    @State private var isHovering: Bool = false
    @State private var isPersonalizing: Bool = false
    @State private var personalized: Bool

    init(document: DocumentItem) {
        self.document = document
        _personalized = State(initialValue: document.isPersonalized)
    }

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
                    Image(systemName: statusIcon)
                        .font(.caption2)
                        .foregroundColor(.secondary)

                    Text(statusText)
                        .font(.caption2)
                        .foregroundColor(.secondary)
                        .lineLimit(1)
                }
                .onHover { inside in
                    withAnimation(.easeInOut(duration: 0.2)) {
                        isHovering = inside
                    }
                }
                .onTapGesture {
                    guard !isPersonalizing else { return }
                    
                    if personalized {
                        // Depersonalize
                        isPersonalizing = true
                        Task {
                            await depersonalizeDocument() // чуть быстрее, чем персонализация
                            withAnimation(.easeInOut(duration: 0.25)) {
                                isPersonalizing = false
                                personalized = false
                            }
                        }
                    } else {
                        // Personalize
                        isPersonalizing = true
                        Task {
                            await personalizeDocument()
                            withAnimation(.easeInOut(duration: 0.25)) {
                                isPersonalizing = false
                                personalized = true
                            }
                        }
                    }
                }
                .animation(.easeInOut(duration: 0.2), value: isHovering)
            }
        }
        .frame(width: 160)
        .contentShape(Rectangle())
    }
    
    private func personalizeDocument() async {
        print("ℹ️ Model context identity: \(ObjectIdentifier(context))")

        let fileURL = document.url.standardizedFileURL // Нормализация пути

        do {
           // Читаем содержимое файла
           let text = try String(contentsOf: fileURL, encoding: .utf8).replacingOccurrences(of: "\n", with: " ")
           
           // Проверяем, что текст не пустой
           guard !text.isEmpty else {
               print("⚠️ Документ пустой, генерация эмбеддинга пропущена: \(fileURL.lastPathComponent)")
               return
           }
           
           print("ℹ️ Начинается генерация эмбеддинга для документа: \(fileURL.lastPathComponent)")
           
           // Используем generateEmbedding для генерации эмбеддингов
           await withCheckedContinuation { continuation in
               llmEngine.generateEmbedding(for: text) { result in
                   switch result {
                   case .success(let embeddings):
                       do {
                           // Кодируем эмбеддинги в Data для хранения в SwiftData
                           let embeddingData = try JSONEncoder().encode(embeddings)
                           
                           // Ищем существующую запись в SwiftData
                           let descriptor = FetchDescriptor<Embedding>(predicate: #Predicate { $0.documentID == fileURL.path })
                           if let existing = try? context.fetch(descriptor).first {
                               // Обновляем существующую запись
                               existing.embeddingData = embeddingData
                               existing.isPersonalized = true
                               do {
                                   try context.save()
                                   print("✅ Successfully saved context for document: \(fileURL.lastPathComponent)")
                               } catch {
                                   print("🫩 Failed to save context: \(error)")
                               }
                               print("✅ Обновлён существующий эмбеддинг для документа: \(fileURL.lastPathComponent)")
                           } else {
                               // Создаём новую запись
                               let embedding = Embedding(
                                   documentID: fileURL.path,
                                   embeddingData: embeddingData,
                                   isPersonalized: true,
                                   documentURL: fileURL
                               )
                               DispatchQueue.main.async {
                                   do {
                                       context.insert(embedding)
                                       try context.save()
                                       print("✅ Successfully saved context for document: \(fileURL.lastPathComponent)")
                                       let allEmbeddings = try? context.fetch(FetchDescriptor<Embedding>())
                                       print("ℹ️ Всего эмбеддингов в базе: \(allEmbeddings?.count ?? 0)")
                                   } catch {
                                       print("🫩 Failed to save context: \(error.localizedDescription)")
                                   }
                               }
                           }
                       } catch {
                           print("🫩 Ошибка сохранения эмбеддингов в SwiftData: \(error)")
                       }
                   case .failure(let error):
                       print("🫩 Ошибка генерации эмбеддингов: \(error)")
                   }
                   continuation.resume()
               }
           }
       } catch {
           print("🫩 Ошибка чтения файла: \(error)")
       }
   }
    
    
    private func depersonalizeDocument() async {
        do {
            let documentID = document.url.path
            // Используем #Predicate с явным указанием типов
            let descriptor = FetchDescriptor<Embedding>(predicate: #Predicate<Embedding> { embedding in
                embedding.documentID == documentID
            })
            if let embedding = try context.fetch(descriptor).first {
                context.delete(embedding)
                try context.save()
                print("✅ Эмбеддинг удалён из SwiftData для документа: \(document.url.lastPathComponent)")
            } else {
                print("ℹ️ Эмбеддинг для документа \(document.url.lastPathComponent) не найден")
            }
        } catch {
            print("🫩 Ошибка удаления эмбеддинга: \(error)")
        }
    }

    // MARK: - Computed Properties
    private var statusIcon: String {
        if isPersonalizing {
            return personalized ? "arrow.triangle.2.circlepath.circle" : "arrow.triangle.2.circlepath"
        }
        if personalized {
            return isHovering ? "arrow.uturn.backward.circle" : "checkmark.circle.fill"
        } else {
            return isHovering ? "checkmark.circle" : "xmark.circle.fill"
        }
    }

    private var statusText: String {
        if isPersonalizing {
            return personalized ? "Сброс..." : "Персонализация..."
        }
        if personalized {
            return isHovering ? "Сбросить?" : "Персонализирован"
        } else {
            return isHovering ? "Персонализировать?" : "Не персонализирован"
        }
    }
}

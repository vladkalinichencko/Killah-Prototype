import SwiftUI

/// Floating overlay that appears at the bottom-center while the LLM engine is starting.
struct LoadingOverlayView: View {
    var body: some View {
        HStack(spacing: 8) {
            ProgressView()
                .controlSize(.small)
            Text("Загружаем модель…")
                .font(.system(size: 14, weight: .medium))
        }
        .padding(.horizontal, 16)
        .padding(.vertical, 12)
        .background(.thinMaterial)
        .clipShape(Capsule())
        .shadow(radius: 3)
    }
}

#Preview {
    LoadingOverlayView()
} 
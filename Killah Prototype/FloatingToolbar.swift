import SwiftUI
import AppKit

/// Floating toolbar with formatting controls
struct FloatingToolbar: View {
    weak var formattingDelegate: TextFormattingDelegate?
    var isBoldActive: Bool
    var isItalicActive: Bool
    var isUnderlineActive: Bool
    var isStrikethroughActive: Bool
    private let fontManager = FontManager.shared
    
    var body: some View {
        HStack(spacing: 12) {
            // Text formatting group
            HStack(spacing: 8) {
                Button(action: { formattingDelegate?.toggleBold() }) {
                    Image(systemName: "bold")
                        .font(.system(size: fontManager.toolbarIconSize, weight: .medium))
                        .foregroundColor(isBoldActive ? .accentColor : .primary)
                }
                .buttonStyle(ToolbarButtonStyle(isActive: isBoldActive))

                Button(action: { formattingDelegate?.toggleItalic() }) {
                    Image(systemName: "italic")
                        .font(.system(size: fontManager.toolbarIconSize, weight: .medium))
                        .foregroundColor(isItalicActive ? .accentColor : .primary)
                }
                .buttonStyle(ToolbarButtonStyle(isActive: isItalicActive))

                Button(action: { formattingDelegate?.toggleUnderline() }) {
                    Image(systemName: "underline")
                        .font(.system(size: fontManager.toolbarIconSize, weight: .medium))
                        .foregroundColor(isUnderlineActive ? .accentColor : .primary)
                }
                .buttonStyle(ToolbarButtonStyle(isActive: isUnderlineActive))

                Button(action: { formattingDelegate?.toggleStrikethrough() }) {
                    Image(systemName: "strikethrough")
                        .font(.system(size: fontManager.toolbarIconSize, weight: .medium))
                        .foregroundColor(isStrikethroughActive ? .accentColor : .primary)
                }
                .buttonStyle(ToolbarButtonStyle(isActive: isStrikethroughActive))
            }
            
            Divider().frame(height: 20)
            
            // List formatting
            HStack(spacing: 8) {
                Button(action: { formattingDelegate?.toggleBulletList() }) {
                    Image(systemName: "list.bullet")
                        .font(.system(size: fontManager.toolbarIconSize, weight: .medium))
                        .foregroundColor(.primary)
                }
                .buttonStyle(ToolbarButtonStyle())
                
                Button(action: { formattingDelegate?.toggleNumberedList() }) {
                    Image(systemName: "list.number")
                        .font(.system(size: fontManager.toolbarIconSize, weight: .medium))
                        .foregroundColor(.primary)
                }
                .buttonStyle(ToolbarButtonStyle())
            }
            
            Divider().frame(height: 20)
            
            // Alignment
            HStack(spacing: 8) {
                Button(action: { formattingDelegate?.setTextAlignment(.left) }) { Image(systemName: "text.alignleft").font(.system(size: fontManager.toolbarIconSize, weight: .medium)).foregroundColor(.primary) }
                    .buttonStyle(ToolbarButtonStyle())
                Button(action: { formattingDelegate?.setTextAlignment(.center) }) { Image(systemName: "text.aligncenter").font(.system(size: fontManager.toolbarIconSize, weight: .medium)).foregroundColor(.primary) }
                    .buttonStyle(ToolbarButtonStyle())
                Button(action: { formattingDelegate?.setTextAlignment(.right) }) { Image(systemName: "text.alignright").font(.system(size: fontManager.toolbarIconSize, weight: .medium)).foregroundColor(.primary) }
                    .buttonStyle(ToolbarButtonStyle())
            }
            
            Divider().frame(height: 20)
            
            // Font and highlight
            HStack(spacing: 8) {
                Button(action: { formattingDelegate?.decreaseFontSize() }) { Image(systemName: "textformat.size.smaller").font(.system(size: fontManager.toolbarIconSize, weight: .medium)).foregroundColor(.primary) }
                    .buttonStyle(ToolbarButtonStyle())
                Button(action: { formattingDelegate?.increaseFontSize() }) { Image(systemName: "textformat.size.larger").font(.system(size: fontManager.toolbarIconSize, weight: .medium)).foregroundColor(.primary) }
                    .buttonStyle(ToolbarButtonStyle())
                Button(action: { formattingDelegate?.toggleHighlight() }) { Image(systemName: "highlighter").font(.system(size: fontManager.toolbarIconSize, weight: .medium)).foregroundColor(.primary) }
                    .buttonStyle(ToolbarButtonStyle())

                Divider().frame(height: 20)
                
                Button(action: { 
                    openFontPanel()
                }) {
                    Image(systemName: "textformat.abc").font(.system(size: fontManager.toolbarIconSize, weight: .medium)).foregroundColor(.primary)
                }
                .buttonStyle(ToolbarButtonStyle())
            }
        }
        .padding(.horizontal, 16)
        .padding(.vertical, 8) // Add vertical padding for better appearance
        .background(.ultraThickMaterial) // Use ultraThickMaterial for a translucent background
        .cornerRadius(8) // Rounded corners for the toolbar
        .shadow(color: Color.black.opacity(0.2), radius: 10, x: 0, y: 2) // Add a shadow
    }
    
    private func openFontPanel() {
        guard NSApp.mainWindow != nil else { return }
        DispatchQueue.main.async {
            NSApp.sendAction(#selector(NSFontManager.orderFrontFontPanel(_:)), to: nil, from: nil)
        }
    }
}

struct ToolbarButtonStyle: ButtonStyle {
    var isActive: Bool = false
    func makeBody(configuration: Configuration) -> some View {
        configuration.label
            .padding(.horizontal, 12)
            .padding(.vertical, 8)
            .background(
                RoundedRectangle(cornerRadius: 8)
                    .fill(isActive ? Color.accentColor.opacity(0.18) : (configuration.isPressed ? Color.primary.opacity(0.15) : Color.clear))
            )
            .scaleEffect(configuration.isPressed ? 0.95 : 1)
            .animation(.easeInOut(duration: 0.1), value: configuration.isPressed)
            .onHover { hovering in
                if hovering {
                    NSCursor.pointingHand.push()
                } else {
                    NSCursor.pop()
                }
            }
    }
}

struct FloatingToolbarVisualEffectView: NSViewRepresentable {
    var material: NSVisualEffectView.Material
    var blendingMode: NSVisualEffectView.BlendingMode

    func makeNSView(context: Context) -> NSVisualEffectView {
        let view = NSVisualEffectView()
        view.material = material
        view.blendingMode = blendingMode
        view.state = .active
        return view
    }

    func updateNSView(_ nsView: NSVisualEffectView, context: Context) {
        nsView.material = material
        nsView.blendingMode = blendingMode
    }
}

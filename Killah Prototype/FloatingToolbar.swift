import SwiftUI
import AppKit

/// Floating toolbar with formatting controls
struct FloatingToolbar: View {
    weak var formattingDelegate: TextFormattingDelegate?
    
    var body: some View {
        HStack(spacing: 12) {
            // Text formatting group
            HStack(spacing: 8) {
                Button(action: {
                    NSApp.sendAction(#selector(NSResponder.toggleBoldface(_:)), to: nil, from: nil)
                }) {
                    Image(systemName: "bold")
                        .font(.system(size: 16, weight: .medium))
                        .foregroundColor(.primary)
                }
                .buttonStyle(ToolbarButtonStyle())

                Button(action: {
                    NSApp.sendAction(#selector(NSResponder.toggleItalics(_:)), to: nil, from: nil)
                }) {
                    Image(systemName: "italic")
                        .font(.system(size: 16, weight: .medium))
                        .foregroundColor(.primary)
                }
                .buttonStyle(ToolbarButtonStyle())

                Button(action: {
                    NSApp.sendAction(#selector(NSResponder.toggleUnderline(_:)), to: nil, from: nil)
                }) {
                    Image(systemName: "underline")
                        .font(.system(size: 16, weight: .medium))
                        .foregroundColor(.primary)
                }
                .buttonStyle(ToolbarButtonStyle())

                Button(action: {
                    NSApp.sendAction(#selector(NSResponder.toggleStrikethrough(_:)), to: nil, from: nil)
                }) {
                    Image(systemName: "strikethrough")
                        .font(.system(size: 16, weight: .medium))
                        .foregroundColor(.primary)
                }
                .buttonStyle(ToolbarButtonStyle())
            }
            
            Divider().frame(height: 20)
            
            // List formatting
            HStack(spacing: 8) {
                Button(action: { formattingDelegate?.toggleBulletList() }) {
                    Image(systemName: "list.bullet")
                        .font(.system(size: 16, weight: .medium))
                        .foregroundColor(.primary)
                }
                .buttonStyle(ToolbarButtonStyle())
                
                Button(action: { formattingDelegate?.toggleNumberedList() }) {
                    Image(systemName: "list.number")
                        .font(.system(size: 16, weight: .medium))
                        .foregroundColor(.primary)
                }
                .buttonStyle(ToolbarButtonStyle())
            }
            
            Divider().frame(height: 20)
            
            // Alignment
            HStack(spacing: 8) {
                Button(action: { formattingDelegate?.setTextAlignment(.left) }) { Image(systemName: "text.alignleft").font(.system(size: 16, weight: .medium)).foregroundColor(.primary) }
                    .buttonStyle(ToolbarButtonStyle())
                Button(action: { formattingDelegate?.setTextAlignment(.center) }) { Image(systemName: "text.aligncenter").font(.system(size: 16, weight: .medium)).foregroundColor(.primary) }
                    .buttonStyle(ToolbarButtonStyle())
                Button(action: { formattingDelegate?.setTextAlignment(.right) }) { Image(systemName: "text.alignright").font(.system(size: 16, weight: .medium)).foregroundColor(.primary) }
                    .buttonStyle(ToolbarButtonStyle())
            }
            
            Divider().frame(height: 20)
            
            // Font and highlight
            HStack(spacing: 8) {
                Button(action: { formattingDelegate?.decreaseFontSize() }) { Image(systemName: "textformat.size.smaller").font(.system(size: 16, weight: .medium)).foregroundColor(.primary) }
                    .buttonStyle(ToolbarButtonStyle())
                Button(action: { formattingDelegate?.increaseFontSize() }) { Image(systemName: "textformat.size.larger").font(.system(size: 16, weight: .medium)).foregroundColor(.primary) }
                    .buttonStyle(ToolbarButtonStyle())
                Button(action: { formattingDelegate?.toggleHighlight() }) { Image(systemName: "highlighter").font(.system(size: 16, weight: .medium)).foregroundColor(.primary) }
                    .buttonStyle(ToolbarButtonStyle())

                Divider().frame(height: 20)
                
                Button(action: { NSApp.sendAction(#selector(NSFontManager.orderFrontFontPanel(_:)), to: nil, from: nil) }) {
                    Image(systemName: "character.font").font(.system(size: 16, weight: .medium)).foregroundColor(.primary)
                }
                .buttonStyle(ToolbarButtonStyle())
            }
        }
        .padding(.horizontal, 16)
        .padding(.vertical, 10)
        .background(VisualEffectView(material: .popover, blendingMode: .withinWindow))
        .cornerRadius(12)
        .shadow(color: Color.black.opacity(0.12), radius: 8, x: 0, y: 4)
        .frame(maxWidth: .infinity)
        .padding(.horizontal, 40)
    }
}

struct ToolbarButtonStyle: ButtonStyle {
    func makeBody(configuration: Configuration) -> some View {
        configuration.label
            .padding(.horizontal, 12)
            .padding(.vertical, 8)
            .background(
                RoundedRectangle(cornerRadius: 8)
                    .fill(configuration.isPressed ? Color.primary.opacity(0.15) : Color.clear)
            )
            .scaleEffect(configuration.isPressed ? 0.95 : 1)
            .animation(.easeInOut(duration: 0.1), value: configuration.isPressed)
    }
}

struct VisualEffectView: NSViewRepresentable {
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

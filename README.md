# 🖋️ Killah - AI-Powered Writing Companion

*Welcome to Killah, where your creativity meets intelligent assistance!*

## ✨ What is Killah?

Killah is a **local-first, offline** writing application designed to augment your creativity, not replace it. Built natively for macOS with Swift and SwiftUI, Killah helps you bring your ideas to life with the assistance of "lil Pushkin" - our custom language model that runs entirely on your machine.

### 🎯 Our Philosophy

We believe that the best writing comes from **you**. Killah doesn't try to write for you - instead, it learns your style, understands your voice, and helps you continue your thoughts when you need that gentle nudge forward. Buy once, own forever - no subscriptions, no internet required, just you, your words, and intelligent assistance that truly belongs to you.

## 🌟 Key Features

- **🧠 Intelligent Caret System**: The text cursor becomes your creative companion, offering contextual suggestions exactly where you need them
- **🎙️ Voice Integration**: Speak your ideas and watch them flow onto the page with real-time voice-to-text
- **📚 Personal Learning**: Killah learns from your writing style to provide personalized suggestions that sound like *you*
- **🌐 Truly Offline**: Works without internet - your AI assistant is always available, wherever you are
- **💾 Smart Document Management**: Organize your work with intelligent file management and quick access to recent documents
- **🎨 Minimalist Design**: Clean, distraction-free interface that lets you focus on what matters - your writing

## 🚀 Getting Started

### Prerequisites

- macOS (Apple Silicon or Intel)
- Xcode 15.0 or later (for development)

### Installation

1. **Clone the repository**

   ```bash
   git clone https://github.com/your-username/killah-prototype.git
   cd killah-prototype
   ```

2. **Open in Xcode**

   ```bash
   open "Killah Prototype.xcodeproj"
   ```

3. **Build and run**
   - Select your target device/simulator
   - Press `Cmd + R` to build and run

### First Launch

When you first open Killah, you'll see the Personal Notes window where you can:

- Create a new document with the "+" button
- Browse your existing documents organized by recent activity
- See personalized vs. non-personalized documents

## 🛠️ For Developers

### Project Structure

```text
Killah Prototype/
├── Killah Prototype/           # Main app source code
│   ├── ContentView.swift       # Main editor interface
│   ├── LLMEngine.swift        # "lil Pushkin" integration
│   ├── InlineSuggestingTextView.swift  # Smart text editor
│   └── TextDocument.swift     # Document model
├── Resources/                  # ML models and Python scripts
│   ├── minillm_export.pt      # Pre-trained model
│   └── autocomplete.py        # ML inference scripts
├── Documents/                  # Technical specifications
└── Scripts/                   # Build and packaging scripts
```

### Key Technologies

- **Swift & SwiftUI**: Native macOS development
- **AppKit**: Advanced text editing capabilities
- **ExecuTorch**: Local ML inference
- **Custom LLM**: "lil Pushkin" model for personalized assistance

### Contributing

We welcome contributions! Whether you're:

- 🐛 Fixing bugs
- ✨ Adding new features
- 📚 Improving documentation
- 🎨 Enhancing the UI/UX

Please feel free to open an issue or submit a pull request.

## 🎯 Development Roadmap

### Current Status (June 2025)

- ✅ Basic text editor with formatting
- ✅ LLM integration foundation
- 🔄 Intelligent caret system (in progress)
- 🔄 Voice input integration (in progress)

### Upcoming Features

- 📝 Advanced document management
- 🎨 Theme customization
- 📊 Writing analytics
- 🔄 Version history for text selections

## 👥 Meet the Team

Killah is being crafted by a passionate team of developers and designers:

- **Vladislav Kalinichenko** - Project Lead & Swift Architecture
- **Polina** - ML Lead & LLM Integration
- **Arthur** - Swift/SwiftUI Development
- **Kira** - ML Engineering & Dataset Preparation
- **Max** - ML Infrastructure & Data Management
- **Zhanna** - UI/UX Design & Visual Assets

## 📖 Documentation

For detailed technical specifications and implementation guides, check out our documentation in the `/Documents` folder:

- [App Technical Specification](Documents/app.md)
- [Project Roles and Timeline](Documents/PROJECT_ROLES_AND_TIMELINE.md)
- [Packaging Guide](Documents/PACKAGING_GUIDE.md)

## 📄 License

This project is proprietary software. All rights reserved.

## 🤝 Connect With Us

Have questions, suggestions, or just want to say hello? We'd love to hear from you!

---

*Made with ❤️ for writers, by writers. Killah - where your words come alive.*

## 🙏 Acknowledgments

Special thanks to the open-source community and the researchers whose work makes projects like Killah possible. We stand on the shoulders of giants.

---

## Happy Writing! 🖋️

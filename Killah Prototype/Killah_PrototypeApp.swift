//
//  Killah_PrototypeApp.swift
//  Killah Prototype
//
//  Created by Владислав Калиниченко on 03.05.2025.
//

import SwiftUI

@main
struct Killah_PrototypeApp: App {
    var body: some Scene {
        WindowGroup {
            ContentView()
                .onAppear {
                    // Ensure the window is available
                    if let window = NSApplication.shared.windows.first {
                        window.titlebarAppearsTransparent = true
                        window.isMovableByWindowBackground = true // Allows dragging the window by its background
                    }
                }
        }
        .windowStyle(.hiddenTitleBar)
        .windowToolbarStyle(.unifiedCompact) // Use unifiedCompact for a more modern look if desired, or remove for default
    }
}

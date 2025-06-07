import AppKit

// Declare formatting selectors so Swift compiler recognizes them
@objc extension NSResponder {
    @objc func toggleBoldface(_ sender: Any?) {}
    @objc func toggleItalics(_ sender: Any?) {}
    @objc func toggleUnderline(_ sender: Any?) {}
    @objc func toggleStrikethrough(_ sender: Any?) {}
}

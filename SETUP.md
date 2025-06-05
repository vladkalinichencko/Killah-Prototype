# Killah Prototype Setup Guide

This guide explains how to set up the Killah Prototype project on a new machine.

## Prerequisites

1. **Python 3.12** from python.org (not Homebrew or system Python)
   - Download from: https://www.python.org/downloads/
   - Install using the official installer
   - This creates `/Library/Frameworks/Python.framework`

2. **Xcode** with command line tools installed

## Setup Process

### 1. Install Python Dependencies

First, install the required Python packages in your local environment for development:

```bash
cd "Killah Prototype/Resources"
pip3 install -r requirements.txt
```

### 2. Build the App

1. Open `Killah Prototype.xcodeproj` in Xcode
2. Build the project (⌘+B)

### 3. Package the App with Python Environment

After building, run the packaging script to bundle Python and dependencies:

```bash
cd Scripts
./package_app.sh
```

This script will:
- Copy Python.framework to the app bundle
- Create a virtual environment inside the app
- Install dependencies from requirements.txt
- Copy Python scripts and models to the app resources
- Fix library paths and code signing

## Troubleshooting

### Error: "Python binary not found"
This means the packaging script hasn't run successfully or Python.framework is missing.

**Solutions:**
1. Ensure Python 3.12 is installed from python.org (not Homebrew)
2. Check that `/Library/Frameworks/Python.framework` exists
3. Run the packaging script: `./Scripts/package_app.sh`
4. Clean and rebuild the project in Xcode

### Error: "Unable to open mach-O" or "fopen failed"
These are PyTorch Metal backend errors in the bundled environment.

**Solutions:**
- The updated Python script now disables Metal backend automatically
- Ensure you're using the latest version of the autocomplete.py script

### Error: "Model file not found"
The minillm_export.pt file is missing from the app bundle.

**Solutions:**
1. Ensure `minillm_export.pt` exists in the `Resources/` folder
2. Run the packaging script which copies it to the app bundle
3. If the model file doesn't exist, you may need to generate it first

## Development Workflow

1. Make changes to Swift code → Build in Xcode
2. Make changes to Python code → Copy manually or rebuild app and run packaging script
3. Test the app

## File Locations in Built App

After packaging, the app structure looks like:
```
YourApp.app/
├── Contents/
│   ├── Frameworks/
│   │   └── Python.framework/          # Python runtime
│   └── Resources/
│       ├── venv/                      # Virtual environment
│       │   └── bin/python3           # Python executable
│       ├── autocomplete.py           # Main Python script
│       ├── minillm_export.pt         # ML model
│       └── requirements.txt          # Dependencies list
```

## Notes

- The app looks for Python resources in the `Resources/` folder within the app bundle
- All paths are resolved at runtime using Bundle.main APIs
- The packaging script handles code signing and library path fixes automatically

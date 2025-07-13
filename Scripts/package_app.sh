#!/bin/bash

echo "🚀 Starting Package Python script"
echo "BUILT_PRODUCTS_DIR: $BUILT_PRODUCTS_DIR"
echo "PRODUCT_NAME: $PRODUCT_NAME"

APP_PATH="$BUILT_PRODUCTS_DIR/$PRODUCT_NAME.app"
RESOURCES_DIR="$APP_PATH/Contents/Resources"
VENV_DST="$RESOURCES_DIR/venv"
PYTHON_VERSION="3.12"
PYTHON_FRAMEWORK_URL="https://github.com/astral-sh/python-build-standalone/releases/download/20250708/cpython-3.12.11+20250708-aarch64-apple-darwin-install_only.tar.gz"

echo "📁 Creating resources directory: $RESOURCES_DIR"
mkdir -p "$RESOURCES_DIR"

PYTHON_FRAMEWORK_LOCAL="/Library/Frameworks/Python.framework"

echo "🔍 Checking for local Python framework at: $PYTHON_FRAMEWORK_LOCAL"
if [ -d "$PYTHON_FRAMEWORK_LOCAL" ]; then
  echo "✅ Found local Python framework, copying..."
  cp -R "$PYTHON_FRAMEWORK_LOCAL" "$RESOURCES_DIR/"
else
  echo "❌ No local Python framework found, downloading..."
  TEMP_DIR=$(mktemp -d)
  echo "📁 Created temp directory: $TEMP_DIR"
  trap "rm -rf $TEMP_DIR" EXIT
  ARCHIVE_PATH="$TEMP_DIR/python-framework.tar.gz"

  echo "⬇️ Downloading Python framework from: $PYTHON_FRAMEWORK_URL"
  curl -L -o "$ARCHIVE_PATH" "$PYTHON_FRAMEWORK_URL"
  if [ $? -eq 0 ]; then
    echo "✅ Download completed successfully"
  else
    echo "❌ Download failed!"
    exit 1
  fi
  
  echo "📁 Extracting archive..."
  cd "$TEMP_DIR"
  tar -xzf "$ARCHIVE_PATH"
  EXTRACTED_FRAMEWORK=$(find . -name "python" -type d | head -1)
  echo "📁 Found extracted framework: $EXTRACTED_FRAMEWORK"
  cp -R -L "$EXTRACTED_FRAMEWORK" "$RESOURCES_DIR/Python.framework"
  echo "✅ Copied framework to: $RESOURCES_DIR/Python.framework"
fi

# Handle different Python framework structures
if [ -d "$RESOURCES_DIR/Python.framework/bin" ]; then
  # Downloaded standalone framework structure
  LOCAL_PYTHON="$RESOURCES_DIR/Python.framework/bin/python3"
  echo "🐍 Using downloaded framework Python: $LOCAL_PYTHON"
  VENV_CREATE_DIR="$RESOURCES_DIR/Python.framework/bin"
else
  # Local system framework structure
  LOCAL_PYTHON="$RESOURCES_DIR/Python.framework/Versions/3.12/bin/python3"
  echo "🐍 Using local framework Python: $LOCAL_PYTHON"
  VENV_CREATE_DIR="$RESOURCES_DIR/Python.framework/Versions/3.12/bin"
fi

echo "🗑️ Removing existing virtual environment: $VENV_DST"
rm -rf "$VENV_DST"

echo "📁 Creating virtual environment..."
cd "$VENV_CREATE_DIR"
"$LOCAL_PYTHON" -m venv "$VENV_DST"
if [ $? -eq 0 ]; then
  echo "✅ Virtual environment created successfully"
else
  echo "❌ Failed to create virtual environment!"
  exit 1
fi

echo "🔧 Activating virtual environment and installing packages..."
source "$VENV_DST/bin/activate"
echo "📦 Installing requirements from: $PROJECT_DIR/Resources/requirements.txt"
pip install -r "$PROJECT_DIR/Resources/requirements.txt"
echo "⬆️ Upgrading pip..."
pip install --upgrade pip
echo "📦 Installing requirements with upgrade strategy..."
pip install --upgrade --upgrade-strategy eager -r "$PROJECT_DIR/Resources/requirements.txt"
echo "🔧 Deactivating virtual environment..."
deactivate

echo "📋 Copying Python scripts to resources..."
cp "$PROJECT_DIR/Resources/autocomplete.py" "$RESOURCES_DIR/"
cp "$PROJECT_DIR/Resources/audio.py" "$RESOURCES_DIR/"
cp "$PROJECT_DIR/Resources/main_llm.py" "$RESOURCES_DIR/"
cp "$PROJECT_DIR/Resources/min_p_sampling.py" "$RESOURCES_DIR/"
cp "$PROJECT_DIR/Resources/requirements.txt" "$RESOURCES_DIR/"
cp "$PROJECT_DIR/Resources/config.env" "$RESOURCES_DIR/"

PYBIN="$VENV_DST/bin/python3"
echo "🔧 Fixing Python binary paths..."
install_name_tool -change "/Library/Frameworks/Python.framework/Versions/$PYTHON_VERSION/lib/libpython${PYTHON_VERSION}.dylib" "@executable_path/../../../Python.framework/lib/libpython${PYTHON_VERSION}.dylib" "$PYBIN" 2>/dev/null || true
codesign --force --sign - "$PYBIN" 2>/dev/null || true

echo "✅ Package Python script completed" 
#!/bin/bash

APP_PATH="$BUILT_PRODUCTS_DIR/$PRODUCT_NAME.app"
RESOURCES_DIR="$APP_PATH/Contents/Resources"
VENV_DST="$RESOURCES_DIR/venv"
PYTHON_VERSION="3.12"
PYTHON_FRAMEWORK_URL="https://github.com/astral-sh/python-build-standalone/releases/download/20250708/cpython-3.12.11+20250708-aarch64-apple-darwin-install_only.tar.gz"

mkdir -p "$RESOURCES_DIR"

PYTHON_FRAMEWORK_LOCAL="/Library/Frameworks/Python.framework"

if [ -d "$PYTHON_FRAMEWORK_LOCAL" ]; then
  cp -R "$PYTHON_FRAMEWORK_LOCAL" "$RESOURCES_DIR/"
else
  TEMP_DIR=$(mktemp -d)
  trap "rm -rf $TEMP_DIR" EXIT
  ARCHIVE_PATH="$TEMP_DIR/python-framework.tar.gz"

  curl -L -o "$ARCHIVE_PATH" "$PYTHON_FRAMEWORK_URL"
  cd "$TEMP_DIR"
  tar -xzf "$ARCHIVE_PATH"
  EXTRACTED_FRAMEWORK=$(find . -name "python" -type d | head -1)
  cp -R -L "$EXTRACTED_FRAMEWORK" "$RESOURCES_DIR/Python.framework"
fi

LOCAL_PYTHON="$RESOURCES_DIR/Python.framework/bin/python3"
rm -rf "$VENV_DST"
cd "$RESOURCES_DIR/Python.framework/bin"
"$LOCAL_PYTHON" -m venv "$VENV_DST"

source "$VENV_DST/bin/activate"
pip install --find-links="$PROJECT_DIR/Resources/wheels" -r "$PROJECT_DIR/Resources/requirements.txt"
pip install --upgrade pip
pip install --upgrade --upgrade-strategy eager --find-links="$PROJECT_DIR/Resources/wheels" -r "$PROJECT_DIR/Resources/requirements.txt"
deactivate

cp "$PROJECT_DIR/Resources/autocomplete.py" "$RESOURCES_DIR/"
cp "$PROJECT_DIR/Resources/requirements.txt" "$RESOURCES_DIR/"

PYBIN="$VENV_DST/bin/python3"
install_name_tool -change "/Library/Frameworks/Python.framework/Versions/$PYTHON_VERSION/lib/libpython${PYTHON_VERSION}.dylib" "@executable_path/../../../Python.framework/lib/libpython${PYTHON_VERSION}.dylib" "$PYBIN" 2>/dev/null || true
codesign --force --sign - "$PYBIN" 2>/dev/null || true 
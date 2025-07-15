#!/bin/bash

echo "🚀 Starting Package Python script"
echo "BUILT_PRODUCTS_DIR: $BUILT_PRODUCTS_DIR"
echo "PRODUCT_NAME: $PRODUCT_NAME"

APP_PATH="$BUILT_PRODUCTS_DIR/$PRODUCT_NAME.app"
RESOURCES_DIR="$APP_PATH/Contents/Resources"
FRAMEWORKS_DIR="$APP_PATH/Contents/Frameworks"
VENV_DST="$RESOURCES_DIR/venv"
PYTHON_VERSION="3.12"
PYTHON_FRAMEWORK_URL="https://github.com/astral-sh/python-build-standalone/releases/download/20250708/cpython-3.12.11+20250708-aarch64-apple-darwin-install_only.tar.gz"
LLAMA_SERVER_LOCAL="$PROJECT_DIR/Resources/llama-server-macos-arm64"
LLAMA_SERVER_URL="https://huggingface.co/poinka/llama-server-macos-arm64/resolve/main/llama-server-macos-arm64"
LLAMA_LIBS_BASE_URL="https://huggingface.co/poinka/llama-server-macos-arm64/resolve/main/libs"
LLAMA_BINARY_CACHE_DIR="$PROJECT_DIR/Resources/llama-binaries"

echo "📁 Creating resources and frameworks directories: $RESOURCES_DIR, $FRAMEWORKS_DIR"
mkdir -p "$RESOURCES_DIR" "$FRAMEWORKS_DIR" "$LLAMA_BINARY_CACHE_DIR"

# Список библиотек, необходимых для llama-server
LIBRARIES=("libmtmd.dylib" "libllama.dylib" "libggml.dylib" "libggml-cpu.dylib" "libggml-blas.dylib" "libggml-metal.dylib" "libggml-base.dylib")

# Проверка и скачивание llama-server
echo "🔍 Checking for llama-server at: $LLAMA_SERVER_LOCAL"
if [ -f "$LLAMA_SERVER_LOCAL" ]; then
  echo "✅ Found cached llama-server, reusing..."
else
  echo "❌ No cached llama-server found, downloading..."
  TEMP_DIR=$(mktemp -d)
  echo "📁 Created temp directory: $TEMP_DIR"
  trap "rm -rf $TEMP_DIR" EXIT
  
  echo "⬇️ Downloading llama-server from: $LLAMA_SERVER_URL"
  curl -L -o "$TEMP_DIR/llama-server-macos-arm64" "$LLAMA_SERVER_URL"
  if [ $? -eq 0 ]; then
    echo "✅ llama-server downloaded successfully"
    mkdir -p "$(dirname "$LLAMA_SERVER_LOCAL")"
    mv "$TEMP_DIR/llama-server-macos-arm64" "$LLAMA_SERVER_LOCAL"
    chmod +x "$LLAMA_SERVER_LOCAL"
  else
    echo "❌ Failed to download llama-server!"
    exit 1
  fi
fi

# Проверка и скачивание библиотек
echo "🔍 Checking for cached libraries at: $LLAMA_BINARY_CACHE_DIR"
ALL_LIBRARIES_CACHED=true
for LIB in "${LIBRARIES[@]}"; do
  if [ ! -f "$LLAMA_BINARY_CACHE_DIR/$LIB" ]; then
    ALL_LIBRARIES_CACHED=false
    break
  fi
done

if [ "$ALL_LIBRARIES_CACHED" = true ]; then
  echo "✅ Found all cached libraries, reusing..."
else
  echo "❌ Some libraries missing, downloading from Hugging Face..."
  TEMP_DIR=$(mktemp -d)
  echo "📁 Created temp directory: $TEMP_DIR"
  trap "rm -rf $TEMP_DIR" EXIT
  
  for LIB in "${LIBRARIES[@]}"; do
    LIB_URL="$LLAMA_LIBS_BASE_URL/$LIB"
    LIB_CACHE_PATH="$LLAMA_BINARY_CACHE_DIR/$LIB"
    echo "⬇️ Downloading $LIB from: $LIB_URL"
    curl -L -o "$TEMP_DIR/$LIB" "$LIB_URL"
    if [ $? -eq 0 ]; then
      echo "✅ $LIB downloaded successfully"
      mkdir -p "$LLAMA_BINARY_CACHE_DIR"
      mv "$TEMP_DIR/$LIB" "$LIB_CACHE_PATH"
      chmod +x "$LIB_CACHE_PATH"
    else
      echo "❌ Failed to download $LIB from $LIB_URL"
      exit 1
    fi
  done
  echo "✅ All libraries cached to $LLAMA_BINARY_CACHE_DIR"
fi

# Проверка и загрузка Python
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

# Обработка структуры Python-фреймворка
if [ -d "$RESOURCES_DIR/Python.framework/bin" ]; then
  LOCAL_PYTHON="$RESOURCES_DIR/Python.framework/bin/python3"
  echo "🐍 Using downloaded framework Python: $LOCAL_PYTHON"
  VENV_CREATE_DIR="$RESOURCES_DIR/Python.framework/bin"
else
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

# Копируем llama-server в виртуальное окружение
echo "📋 Copying llama-server to: $VENV_DST/bin/llama-server"
mkdir -p "$VENV_DST/bin"
cp "$LLAMA_SERVER_LOCAL" "$VENV_DST/bin/llama-server"
if [ $? -eq 0 ]; then
    echo "✅ Successfully copied llama-server"
    ls -l "$VENV_DST/bin/llama-server"
else
    echo "❌ Failed to copy llama-server"
    exit 1
fi

# Копируем библиотеки в Contents/Frameworks/
echo "📋 Copying required libraries to: $FRAMEWORKS_DIR"
for LIB in "${LIBRARIES[@]}"; do
  LIB_SRC="$LLAMA_BINARY_CACHE_DIR/$LIB"
  LIB_DST="$FRAMEWORKS_DIR/$LIB"
  if [ -f "$LIB_SRC" ]; then
    echo "📋 Copying $LIB from $LIB_SRC to $LIB_DST"
    cp "$LIB_SRC" "$LIB_DST"
    if [ $? -eq 0 ]; then
      echo "✅ Successfully copied $LIB"
      chmod +x "$LIB_DST"
      codesign --force --sign - "$LIB_DST"
      if [ $? -eq 0 ]; then
        echo "✅ Successfully signed $LIB"
      else
        echo "❌ Failed to sign $LIB"
        exit 1
      fi
    else
      echo "❌ Failed to copy $LIB from $LIB_SRC"
      exit 1
    fi
  else
    echo "❌ $LIB not found at $LIB_SRC"
    exit 1
  fi
done

# Настраиваем @rpath для библиотек (рекурсивно)
echo "🔧 Configuring @rpath for libraries..."
for LIB in "${LIBRARIES[@]}"; do
  LIB_DST="$FRAMEWORKS_DIR/$LIB"
  if [ -f "$LIB_DST" ]; then
    install_name_tool -add_rpath "@executable_path/../Frameworks" "$LIB_DST"
    for DEP in "${LIBRARIES[@]}"; do
      install_name_tool -change "@rpath/$DEP" "@executable_path/../Frameworks/$DEP" "$LIB_DST"
    done
    if [ $? -eq 0 ]; then
      echo "✅ Updated @rpath for $LIB"
    else
      echo "❌ Failed to update @rpath for $LIB"
      exit 1
    fi
  else
    echo "❌ $LIB not found at $LIB_DST"
    exit 1
  fi
done

# Настраиваем @rpath для llama-server
echo "🔧 Configuring @rpath for llama-server..."
LLAMA_SERVER_BIN="$VENV_DST/bin/llama-server"
install_name_tool -add_rpath "@executable_path/../../../Frameworks" "$LLAMA_SERVER_BIN"
if [ $? -eq 0 ]; then
  echo "✅ Added @rpath for Frameworks"
else
  echo "❌ Failed to add @rpath"
  exit 1
fi

for LIB in "${LIBRARIES[@]}"; do
  install_name_tool -change "@rpath/$LIB" "@executable_path/../../../Frameworks/$LIB" "$LLAMA_SERVER_BIN"
  if [ $? -eq 0 ]; then
    echo "✅ Updated path for $LIB"
  else
    echo "❌ Failed to update path for $LIB"
    exit 1
  fi
done

# Удаляем карантин для всех файлов
echo "🔧 Removing quarantine attributes..."
for LIB in "${LIBRARIES[@]}"; do
  sudo xattr -rd com.apple.quarantine "$FRAMEWORKS_DIR/$LIB" 2>/dev/null || true
done
sudo xattr -rd com.apple.quarantine "$LLAMA_SERVER_BIN" 2>/dev/null || true

# Подписываем llama-server
echo "🔧 Signing llama-server binary..."
codesign --force --sign - "$LLAMA_SERVER_BIN"
if [ $? -eq 0 ]; then
  echo "✅ Successfully signed llama-server"
else
  echo "❌ Failed to sign llama-server"
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
cp "$PROJECT_DIR/Resources/autocomplete.py" "$RESOURCES_DIR/" 2>/dev/null || true
cp "$PROJECT_DIR/Resources/audio.py" "$RESOURCES_DIR/" 2>/dev/null || true
cp "$PROJECT_DIR/Resources/main_llm.py" "$RESOURCES_DIR/" 2>/dev/null || true
cp "$PROJECT_DIR/Resources/min_p_sampling.py" "$RESOURCES_DIR/" 2>/dev/null || true
cp "$PROJECT_DIR/Resources/requirements.txt" "$RESOURCES_DIR/" 2>/dev/null || true
cp "$PROJECT_DIR/Resources/config.env" "$RESOURCES_DIR/" 2>/dev/null || true

PYBIN="$VENV_DST/bin/python3"
echo "🔧 Fixing Python binary paths..."
install_name_tool -change "/Library/Frameworks/Python.framework/Versions/$PYTHON_VERSION/lib/libpython${PYTHON_VERSION}.dylib" "@executable_path/../../../Python.framework/lib/libpython${PYTHON_VERSION}.dylib" "$PYBIN" 2>/dev/null || true
codesign --force --sign - "$PYBIN" 2>/dev/null || true

echo "✅ Package Python script completed"

#!/usr/bin/env bash
set -euo pipefail

# Функция для обработки ошибок
error_handler() {
    echo "❌ ОШИБКА на строке $1: команда '$2' завершилась с кодом $3"
    echo "📍 Отладочная информация на момент ошибки:"
    echo "   PWD: $(pwd)"
    echo "   APP_PATH: ${APP_PATH:-'НЕ УСТАНОВЛЕНА'}"
    echo "   BUILD_DIR: ${BUILD_DIR:-'НЕ УСТАНОВЛЕНА'}"
    exit $3
}

# Подключаем обработчик ошибок
trap 'error_handler ${LINENO} "$BASH_COMMAND" $?' ERR

echo "🚀 УПАКОВКА ПРИЛОЖЕНИЯ"
echo "Время: $(date)"
echo "🔧 Отладочная информация:"
echo "   PWD: $(pwd)"
echo "   BUILT_PRODUCTS_DIR: ${BUILT_PRODUCTS_DIR:-'НЕ УСТАНОВЛЕНА'}"
echo "   PRODUCT_NAME: ${PRODUCT_NAME:-'НЕ УСТАНОВЛЕНА'}"
echo "   SRCROOT: ${SRCROOT:-'НЕ УСТАНОВЛЕНА'}"
echo "   TARGET_NAME: ${TARGET_NAME:-'НЕ УСТАНОВЛЕНА'}"
echo "   PROJECT_DIR will be: $(cd "$(dirname "$0")/.." && pwd)"

# ===================================================================
# КОНФИГУРАЦИЯ - единственное место где задаются все пути и настройки
# ===================================================================

PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
APP_NAME="${PRODUCT_NAME:-Killah Prototype}.app"
PYTHON_VERSION="3.12"  # По умолчанию, но будет автоопределена позже
VENV_NAME="venv"
# URL для скачивания предварительно собранного Python.framework
PYTHON_FRAMEWORK_URL="https://github.com/python/cpython-bin-deps/releases/download/20231002/cpython-3.12.0%2B20231002-x86_64-apple-darwin-install_only.tar.gz"
PYTHON_FRAMEWORK_LOCAL="/Library/Frameworks/Python.framework"
MODEL_FILE_NAME="gemma-3-4b-pt-q4_0.gguf" # <--- ДОБАВЛЕНО: Имя файла модели

# Пути сборки из Xcode (с резервными значениями для отладки)
if [ -z "${BUILT_PRODUCTS_DIR:-}" ]; then
  echo "⚠️  BUILT_PRODUCTS_DIR не установлена, используем значение по умолчанию"
  BUILD_DIR="$PROJECT_DIR/build/Debug"
else
  BUILD_DIR="$BUILT_PRODUCTS_DIR"
fi

APP_PATH="$BUILD_DIR/$APP_NAME"
FRAMEWORKS_DIR="$APP_PATH/Contents/Frameworks"
RESOURCES_DIR="$APP_PATH/Contents/Resources"
VENV_DST="$RESOURCES_DIR/$VENV_NAME"

echo "📁 Приложение: $APP_PATH"

# ===================================================================
# ПРОВЕРКИ И ПОДГОТОВКА
# ===================================================================

# Проверяем существование .app
if [ ! -d "$APP_PATH" ]; then
  echo "❌ .app не найден: $APP_PATH"
  exit 1
fi
echo "✅ .app найден"

# Создаем необходимые папки
mkdir -p "$FRAMEWORKS_DIR"
mkdir -p "$RESOURCES_DIR"

# ===================================================================
# ПОЛУЧЕНИЕ PYTHON.FRAMEWORK
# ===================================================================

get_python_framework() {
  local framework_dst="$FRAMEWORKS_DIR/Python.framework"
  
  if [ -d "$PYTHON_FRAMEWORK_LOCAL" ]; then
    echo "📋 Копируем локальный Python.framework..."
    
    # Умное копирование: сначала пытаемся с разыменованием символических ссылок
    echo "🔗 Пытаемся скопировать с разыменованием символических ссылок..."
    if cp -R -L "$PYTHON_FRAMEWORK_LOCAL" "$FRAMEWORKS_DIR/" 2>/dev/null; then
      echo "✅ Локальный Python.framework скопирован с полным разыменованием ссылок"
    else
      echo "⚠️  Не удалось скопировать с разыменованием (битые симлинки), пробуем без разыменования..."
      # Удаляем частично скопированный фреймворк если он есть
      [ -d "$framework_dst" ] && rm -rf "$framework_dst"
      
      if cp -R -P "$PYTHON_FRAMEWORK_LOCAL" "$FRAMEWORKS_DIR/" 2>/dev/null; then
        echo "✅ Локальный Python.framework скопирован без разыменования ссылок"
      else
        echo "❌ Стандартное копирование не работает, используем rsync или выборочное копирование..."
        
        # Попробуем rsync если доступен
        if command -v rsync >/dev/null 2>&1; then
          echo "🔄 Пытаемся rsync..."
          if rsync -av --exclude='*/PrivateHeaders' "$PYTHON_FRAMEWORK_LOCAL/" "$framework_dst/" 2>/dev/null; then
            echo "✅ Python.framework скопирован через rsync (исключены проблемные PrivateHeaders)"
          else
            echo "❌ rsync тоже не работает, делаем выборочное копирование..."
            manual_copy_framework "$PYTHON_FRAMEWORK_LOCAL" "$framework_dst"
          fi
        else
          echo "📁 rsync недоступен, делаем выборочное копирование..."
          manual_copy_framework "$PYTHON_FRAMEWORK_LOCAL" "$framework_dst"
        fi
      fi
    fi
  else
    echo "⚠️  Локальный Python.framework не найден: $PYTHON_FRAMEWORK_LOCAL"
    echo "📥 Скачиваем предварительно собранный Python.framework..."
    
    # Создаем временную папку для скачивания
    local temp_dir=$(mktemp -d)
    trap "rm -rf $temp_dir" EXIT
    
    local archive_path="$temp_dir/python-framework.tar.gz"
    
    echo "🌐 Скачиваем с: $PYTHON_FRAMEWORK_URL"
    if curl -L -o "$archive_path" "$PYTHON_FRAMEWORK_URL"; then
      echo "✅ Архив скачан"
      
      echo "📦 Распаковываем Python.framework..."
      cd "$temp_dir"
      tar -xzf "$archive_path"
      
      # Ищем Python.framework в распакованном архиве
      local extracted_framework=$(find . -name "Python.framework" -type d | head -1)
      if [ -n "$extracted_framework" ]; then
        echo "📋 Копируем распакованный Python.framework..."
        
        # Применяем ту же умную логику копирования для скачанного фреймворка
        if cp -R -L "$extracted_framework" "$FRAMEWORKS_DIR/" 2>/dev/null; then
          echo "✅ Python.framework из архива скопирован с полным разыменованием ссылок"
        else
          echo "⚠️  Не удалось скопировать с разыменованием, пробуем без разыменования..."
          # Удаляем частично скопированный фреймворк если он есть
          [ -d "$framework_dst" ] && rm -rf "$framework_dst"
          
          if cp -R -P "$extracted_framework" "$FRAMEWORKS_DIR/" 2>/dev/null; then
            echo "✅ Python.framework из архива скопирован без разыменования ссылок"
          else
            echo "🔄 Используем выборочное копирование..."
            manual_copy_framework "$extracted_framework" "$framework_dst"
          fi
        fi
      else
        echo "❌ Python.framework не найден в архиве"
        exit 1
      fi
    else
      echo "❌ Не удалось скачать Python.framework"
      echo "💡 Установите Python.framework локально или обновите URL"
      exit 1
    fi
  fi
}

# Функция для выборочного копирования фреймворка
manual_copy_framework() {
  local src="$1"
  local dst="$2"
  
  echo "🔧 Выполняем выборочное копирование фреймворка..."
  mkdir -p "$dst"
  
  # Копируем содержимое фреймворка, проверяя каждый элемент
  for item in "$src"/*; do
    if [ -e "$item" ] || [ -L "$item" ]; then
      local basename=$(basename "$item")
      echo "  📁 Проверяем: $basename"
      
      if [ -L "$item" ]; then
        # Для символических ссылок проверяем, существует ли цель
        if [ -e "$item" ]; then
          echo "    ✅ Валидная символическая ссылка, копируем с разыменованием"
          cp -R -L "$item" "$dst/" 2>/dev/null || {
            echo "    ⚠️  Не удалось разыменовать, копируем как ссылку"
            cp -R -P "$item" "$dst/"
          }
        else
          echo "    ⚠️  Битая символическая ссылка ($basename), пропускаем"
        fi
      else
        echo "    ✅ Обычный файл/папка, копируем"
        cp -R "$item" "$dst/"
      fi
    fi
  done
  echo "✅ Выборочное копирование завершено"
}

get_python_framework

# Автоопределение версии Python
echo "📋 Определяем версию Python..."
ACTUAL_PYTHON_VERSION=$(ls "$FRAMEWORKS_DIR/Python.framework/Versions/" | grep -E "^[0-9]+\.[0-9]+$" | head -1)
if [ -n "$ACTUAL_PYTHON_VERSION" ]; then
  PYTHON_VERSION="$ACTUAL_PYTHON_VERSION"
  echo "✅ Обнаружена версия Python: $PYTHON_VERSION"
else
  echo "⚠️  Не удалось автоопределить версию Python, используем: $PYTHON_VERSION"
fi

# ===================================================================
# СОЗДАНИЕ ВИРТУАЛЬНОГО ОКРУЖЕНИЯ
# ===================================================================

echo "📋 Создаем venv..."
PYTHON_BIN="$FRAMEWORKS_DIR/Python.framework/Versions/$PYTHON_VERSION/bin/python3"

if [ ! -f "$PYTHON_BIN" ]; then
  echo "❌ Python binary не найден: $PYTHON_BIN"
  echo "💡 Доступные версии: $(ls "$FRAMEWORKS_DIR/Python.framework/Versions/" || echo "нет")"
  exit 1
fi

"$PYTHON_BIN" -m venv "$VENV_DST"
echo "✅ venv создан"

# ===================================================================
# УСТАНОВКА ЗАВИСИМОСТЕЙ
# ===================================================================

echo "📋 Устанавливаем зависимости..."
source "$VENV_DST/bin/activate"

echo "🐍 Скачиваем пакеты в локальный кэш (если их нет)..."
pip download -r "$PROJECT_DIR/Resources/requirements.txt" -d "$PROJECT_DIR/Resources/wheels"

echo "💿 Устанавливаем пакеты из локального кэша..."
pip install --no-index --find-links="$PROJECT_DIR/Resources/wheels" -r "$PROJECT_DIR/Resources/requirements.txt"

deactivate
echo "✅ Зависимости установлены"

# ===================================================================
# КОПИРОВАНИЕ РЕСУРСОВ
# ===================================================================

echo "🖼️  Копируем ресурсы..."

# Копируем модель LLM
echo "🧠 Копируем модель LLM: $MODEL_FILE_NAME"
cp "$PROJECT_DIR/Resources/$MODEL_FILE_NAME" "$RESOURCES_DIR/"
echo "✅ Модель LLM скопирована"

# Копируем скрипт автодополнения
echo "🐍 Копируем скрипт автодополнения..."
cp "$PROJECT_DIR/Resources/autocomplete.py" "$RESOURCES_DIR/"
echo "✅ Скрипт автодополнения скопирован"

# Копируем requirements.txt для Python
echo "📋 Копируем requirements.txt..."
cp "$PROJECT_DIR/Resources/requirements.txt" "$RESOURCES_DIR/"
echo "✅ requirements.txt скопирован"

# ===================================================================
# ИСПРАВЛЕНИЕ ПУТЕЙ И ПОДПИСЬ
# ===================================================================

echo "📋 Исправляем пути библиотек..."
PYBIN="$VENV_DST/bin/python3"

# Исправляем пути библиотек (подавляем предупреждения о подписи)
install_name_tool -change \
  "/Library/Frameworks/Python.framework/Versions/$PYTHON_VERSION/lib/libpython${PYTHON_VERSION}.dylib" \
  "@executable_path/../../../Frameworks/Python.framework/Versions/$PYTHON_VERSION/lib/libpython${PYTHON_VERSION}.dylib" \
  "$PYBIN" 2>/dev/null || true
echo "✅ Пути исправлены"

echo "📋 Переподписываем python3..."
codesign --force --sign - "$PYBIN" 2>/dev/null || true
echo "✅ Переподписано"


# ===================================================================
# ФИНАЛИЗАЦИЯ
# ===================================================================

echo "🎉 УПАКОВКА ЗАВЕРШЕНА УСПЕШНО!"
echo "📊 Статистика:"
echo "   Размер приложения: $(du -sh "$APP_PATH" 2>/dev/null || echo "Не удалось определить")"
echo "   Python.framework: $(du -sh "$FRAMEWORKS_DIR/Python.framework" 2>/dev/null || echo "Не удалось определить")"
echo "   venv: $(du -sh "$VENV_DST" 2>/dev/null || echo "Не удалось определить")"
echo "   Время завершения: $(date)"

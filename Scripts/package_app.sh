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
echo "   PROJECT_DIR will be: $(cd "$(dirname "$0")/.." && pwd)"

# ===================================================================
# КОНФИГУРАЦИЯ - единственное место где задаются все пути и настройки
# ===================================================================

PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
APP_NAME="Killah Prototype.app"
PYTHON_VERSION="3.12"
VENV_NAME="venv"

# URL для скачивания предварительно собранного Python.framework
PYTHON_FRAMEWORK_URL="https://github.com/python/cpython-bin-deps/releases/download/20231002/cpython-3.12.0%2B20231002-x86_64-apple-darwin-install_only.tar.gz"
PYTHON_FRAMEWORK_LOCAL="/Library/Frameworks/Python.framework"

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
    cp -R -L "$PYTHON_FRAMEWORK_LOCAL" "$FRAMEWORKS_DIR/"
    echo "✅ Локальный Python.framework скопирован"
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
        cp -R -L "$extracted_framework" "$FRAMEWORKS_DIR/"
        echo "✅ Python.framework из архива скопирован"
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

get_python_framework

# ===================================================================
# СОЗДАНИЕ ВИРТУАЛЬНОГО ОКРУЖЕНИЯ
# ===================================================================

echo "📋 Создаем venv..."
PYTHON_BIN="$FRAMEWORKS_DIR/Python.framework/Versions/$PYTHON_VERSION/bin/python3"

if [ ! -f "$PYTHON_BIN" ]; then
  echo "❌ Python binary не найден: $PYTHON_BIN"
  exit 1
fi

"$PYTHON_BIN" -m venv "$VENV_DST"
echo "✅ venv создан"

# ===================================================================
# УСТАНОВКА ЗАВИСИМОСТЕЙ
# ===================================================================

echo "📋 Устанавливаем зависимости..."
source "$VENV_DST/bin/activate"
pip install --upgrade pip
pip install -r "$PROJECT_DIR/Resources/requirements.txt"
deactivate
echo "✅ Зависимости установлены"

# ===================================================================
# КОПИРОВАНИЕ РЕСУРСОВ
# ===================================================================

echo "📋 Копируем Python файлы..."
cp "$PROJECT_DIR/Resources/autocomplete.py" "$RESOURCES_DIR/"
cp "$PROJECT_DIR/Resources/minillm_export.pt" "$RESOURCES_DIR/"
cp "$PROJECT_DIR/Resources/requirements.txt" "$RESOURCES_DIR/"
echo "✅ Python файлы скопированы"

# ===================================================================
# ИСПРАВЛЕНИЕ ПУТЕЙ И ПОДПИСЬ
# ===================================================================

echo "📋 Исправляем пути библиотек..."
PYBIN="$VENV_DST/bin/python3"
install_name_tool -change \
  "/Library/Frameworks/Python.framework/Versions/$PYTHON_VERSION/lib/libpython${PYTHON_VERSION}.dylib" \
  "@executable_path/../../../Frameworks/Python.framework/Versions/$PYTHON_VERSION/lib/libpython${PYTHON_VERSION}.dylib" \
  "$PYBIN"
echo "✅ Пути исправлены"

echo "📋 Переподписываем python3..."
codesign --force --sign - "$PYBIN"
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

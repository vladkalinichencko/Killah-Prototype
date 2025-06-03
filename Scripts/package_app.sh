#!/usr/bin/env bash
set -euo pipefail

# Параметры
PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"  # Корень проекта
APP_NAME="Killah Prototype.app"                   # Имя приложения
PYTHON_VERSION="3.12"                            # Версия Python
PYTHON_FRAMEWORK_SRC="/Library/Frameworks/Python.framework"  # Исходный Python.framework
VENV_NAME="venv"                                 # Имя виртуального окружения

# Путь к папке сборки из переменной Xcode
BUILD_DIR="$BUILT_PRODUCTS_DIR"                  # Используем встроенную переменную Xcode
APP_PATH="$BUILD_DIR/$APP_NAME"                  # Полный путь к .app

# Проверяем, что .app существует
if [ ! -d "$APP_PATH" ]; then
  echo "❌ .app не найден по пути $APP_PATH"
  exit 1
fi

FRAMEWORKS_DIR="$APP_PATH/Contents/Frameworks"   # Путь к Frameworks
RESOURCES_DIR="$APP_PATH/Contents/Resources"     # Путь к Resources
VENV_DST="$RESOURCES_DIR/$VENV_NAME"             # Путь к venv

echo "⏳ Старт упаковки .app..."

# Проверяем наличие Python.framework
if [ ! -d "$PYTHON_FRAMEWORK_SRC" ]; then
  echo "❌ Python.framework не найден в $PYTHON_FRAMEWORK_SRC. Установите Python 3.12 с python.org."
  exit 1
fi

# Копируем Python.framework с разыменованием ссылок
echo "→ Копируем Python.framework"
mkdir -p "$FRAMEWORKS_DIR"
cp -R -L "$PYTHON_FRAMEWORK_SRC" "$FRAMEWORKS_DIR/" || {  # -L разыменовывает ссылки
  echo "❌ Не удалось скопировать Python.framework"
  exit 1
}

# Создаём venv
echo "→ Создаём virtualenv"
PYTHON_BIN="$FRAMEWORKS_DIR/Python.framework/Versions/$PYTHON_VERSION/bin/python3"
if [ ! -f "$PYTHON_BIN" ]; then
  echo "❌ Python бинарник не найден в $PYTHON_BIN"
  exit 1
fi
"$PYTHON_BIN" -m venv "$VENV_DST" || {
  echo "❌ Не удалось создать venv"
  exit 1
}

# Устанавливаем зависимости
echo "→ Устанавливаем зависимости из requirements.txt"
source "$VENV_DST/bin/activate"
pip install -r "$PROJECT_DIR/Resources/requirements.txt" || {
  echo "❌ Не удалось установить зависимости"
  exit 1
}
deactivate

# Патчим пути
echo "→ Патчим пути в python3"
PYBIN="$VENV_DST/bin/python3"
install_name_tool -change \
  "/Library/Frameworks/Python.framework/Versions/$PYTHON_VERSION/lib/libpython${PYTHON_VERSION}.dylib" \
  "@executable_path/../../../Frameworks/Python.framework/Versions/$PYTHON_VERSION/lib/libpython${PYTHON_VERSION}.dylib" \
  "$PYBIN" || {  # Исправленный путь: три уровня вверх до Contents, затем в Frameworks
  echo "❌ Не удалось патчить libpython"
  exit 1
}

# Переподписываем python3
echo "→ Переподписываем python3"
codesign --force --sign - "$PYBIN" || {
  echo "❌ Не удалось переподписать python3"
  exit 1
}

echo "✅ Упаковка .app завершена!"

#!/usr/bin/env bash
set -euo pipefail

echo "🚀 УПАКОВКА ПРИЛОЖЕНИЯ"
echo "Время: $(date)"

# Четкие пути
PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
APP_NAME="Killah Prototype.app"
PYTHON_VERSION="3.12"
PYTHON_FRAMEWORK_SRC="/Library/Frameworks/Python.framework"
VENV_NAME="venv"

# Пути сборки
BUILD_DIR="$BUILT_PRODUCTS_DIR"
APP_PATH="$BUILD_DIR/$APP_NAME"
FRAMEWORKS_DIR="$APP_PATH/Contents/Frameworks"
RESOURCES_DIR="$APP_PATH/Contents/Resources"
VENV_DST="$RESOURCES_DIR/$VENV_NAME"

echo "📁 Приложение: $APP_PATH"

# Проверка приложения
if [ ! -d "$APP_PATH" ]; then
  echo "❌ .app не найден: $APP_PATH"
  exit 1
fi

echo "✅ .app найден"

# Создаем папки
mkdir -p "$FRAMEWORKS_DIR"
mkdir -p "$RESOURCES_DIR"

echo "📋 Копируем Python.framework..."
cp -R -L "$PYTHON_FRAMEWORK_SRC" "$FRAMEWORKS_DIR/"
echo "✅ Python.framework скопирован"

echo "📋 Создаем venv..."
PYTHON_BIN="$FRAMEWORKS_DIR/Python.framework/Versions/$PYTHON_VERSION/bin/python3"
"$PYTHON_BIN" -m venv "$VENV_DST"
echo "✅ venv создан"

echo "📋 Устанавливаем зависимости..."
source "$VENV_DST/bin/activate"
pip install -r "$PROJECT_DIR/Resources/requirements.txt"
deactivate
echo "✅ Зависимости установлены"

echo "📋 Копируем Python файлы..."
cp "$PROJECT_DIR/Resources/autocomplete.py" "$RESOURCES_DIR/"
cp "$PROJECT_DIR/Resources/minillm_export.pt" "$RESOURCES_DIR/"
cp "$PROJECT_DIR/Resources/requirements.txt" "$RESOURCES_DIR/"
echo "✅ Python файлы скопированы"

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

echo "🎉 УПАКОВКА ЗАВЕРШЕНА УСПЕШНО!"
echo "   PYTHON_FRAMEWORK_SRC: $PYTHON_FRAMEWORK_SRC"
echo "   VENV_NAME: $VENV_NAME"

# Путь к папке сборки из переменной Xcode
BUILD_DIR="$BUILT_PRODUCTS_DIR"                  # Используем встроенную переменную Xcode
APP_PATH="$BUILD_DIR/$APP_NAME"                  # Полный путь к .app

echo "🎯 ПУТИ СБОРКИ:"
echo "   BUILD_DIR: $BUILD_DIR"
echo "   APP_PATH: $APP_PATH"
echo "   BUILT_PRODUCTS_DIR env var: ${BUILT_PRODUCTS_DIR:-НЕ ЗАДАНО}"

# Проверяем, что .app существует
echo "🔍 ПРОВЕРЯЕМ СУЩЕСТВОВАНИЕ .app..."
if [ ! -d "$APP_PATH" ]; then
  echo "❌❌❌ .app НЕ НАЙДЕН ПО ПУТИ $APP_PATH"
  echo "📁 СОДЕРЖИМОЕ BUILD_DIR ($BUILD_DIR):"
  if [ -d "$BUILD_DIR" ]; then
    ls -la "$BUILD_DIR" || echo "Ошибка чтения BUILD_DIR"
  else
    echo "BUILD_DIR не существует!"
  fi
  exit 1
else
  echo "✅ .app НАЙДЕН: $APP_PATH"
fi

FRAMEWORKS_DIR="$APP_PATH/Contents/Frameworks"   # Путь к Frameworks
RESOURCES_DIR="$APP_PATH/Contents/Resources"     # Путь к Resources
VENV_DST="$RESOURCES_DIR/$VENV_NAME"             # Путь к venv

echo "📂 СТРУКТУРА ПРИЛОЖЕНИЯ:"
echo "   FRAMEWORKS_DIR: $FRAMEWORKS_DIR"
echo "   RESOURCES_DIR: $RESOURCES_DIR"
echo "   VENV_DST: $VENV_DST"

echo "⏳ Старт упаковки .app..."

# Умная проверка - пропускаем упаковку если все уже готово
echo "🔍 ПРОВЕРЯЕМ, НУЖНА ЛИ УПАКОВКА..."

# Проверяем, есть ли уже все необходимое
NEED_PACKAGING=false

# Проверяем Python.framework
if [ ! -d "$FRAMEWORKS_DIR/Python.framework" ]; then
    echo "   ❌ Python.framework отсутствует - нужна упаковка"
    NEED_PACKAGING=true
fi

# Проверяем venv
if [ ! -d "$VENV_DST" ] || [ ! -f "$VENV_DST/bin/python3" ]; then
    echo "   ❌ venv отсутствует или поврежден - нужна упаковка"
    NEED_PACKAGING=true
fi

# Проверяем torch в venv
if [ -f "$VENV_DST/bin/python3" ]; then
    if ! "$VENV_DST/bin/python3" -c "import torch" 2>/dev/null; then
        echo "   ❌ torch не установлен в venv - нужна упаковка"
        NEED_PACKAGING=true
    fi
fi

# Проверяем файлы ресурсов
if [ ! -f "$RESOURCES_DIR/autocomplete.py" ]; then
    echo "   ❌ autocomplete.py отсутствует - нужно копирование"
    NEED_PACKAGING=true
fi

if [ ! -f "$RESOURCES_DIR/minillm_export.pt" ]; then
    echo "   ❌ minillm_export.pt отсутствует - нужно копирование"
    NEED_PACKAGING=true
fi

if [ "$NEED_PACKAGING" = false ]; then
    echo "✅ ВСЕ УЖЕ ГОТОВО! Упаковка не требуется."
    echo "🎯 Приложение содержит:"
    echo "   ✅ Python.framework"
    echo "   ✅ venv с установленными зависимостями"
    echo "   ✅ Все необходимые файлы"
    echo "✅ Упаковка завершена (ничего не изменено)!"
    exit 0
else
    echo "🚀 НАЧИНАЕМ УПАКОВКУ..."
fi

# Проверяем наличие Python.framework
echo "🐍 ПРОВЕРЯЕМ PYTHON.FRAMEWORK..."
echo "   Ожидаемый путь: $PYTHON_FRAMEWORK_SRC"
if [ ! -d "$PYTHON_FRAMEWORK_SRC" ]; then
  echo "❌❌❌ Python.framework НЕ НАЙДЕН в $PYTHON_FRAMEWORK_SRC"
  echo "🔍 ПРОВЕРЯЕМ АЛЬТЕРНАТИВНЫЕ МЕСТА:"
  
  for py_path in "/Library/Frameworks/Python.framework" "/System/Library/Frameworks/Python.framework" "/usr/local/Frameworks/Python.framework"; do
    echo "   Проверяем: $py_path"
    if [ -d "$py_path" ]; then
      echo "   ✅ НАЙДЕН: $py_path"
      ls -la "$py_path/Versions/" 2>/dev/null || echo "   Ошибка чтения версий"
    else
      echo "   ❌ НЕ НАЙДЕН: $py_path"
    fi
  done
  
  echo "💡 УСТАНОВИТЕ Python 3.12 с python.org"
  echo "💡 Ссылка: https://www.python.org/downloads/"
  exit 1
else
  echo "✅ Python.framework НАЙДЕН: $PYTHON_FRAMEWORK_SRC"
  echo "📁 ВЕРСИИ Python:"
  ls -la "$PYTHON_FRAMEWORK_SRC/Versions/" 2>/dev/null || echo "Ошибка чтения версий"
fi

# Копируем Python.framework с разыменованием ссылок
echo "→ КОПИРУЕМ Python.framework..."
echo "   ИЗ: $PYTHON_FRAMEWORK_SRC"
echo "   В: $FRAMEWORKS_DIR/"

mkdir -p "$FRAMEWORKS_DIR"
echo "✅ Создана папка Frameworks: $FRAMEWORKS_DIR"

echo "🔄 КОПИРОВАНИЕ Python.framework (это может занять время)..."
cp -R -L "$PYTHON_FRAMEWORK_SRC" "$FRAMEWORKS_DIR/" || {
  echo "❌❌❌ НЕ УДАЛОСЬ СКОПИРОВАТЬ Python.framework"
  echo "🔍 ОТЛАДОЧНАЯ ИНФОРМАЦИЯ:"
  echo "   Исходная папка существует: $([ -d "$PYTHON_FRAMEWORK_SRC" ] && echo "ДА" || echo "НЕТ")"
  echo "   Целевая папка существует: $([ -d "$FRAMEWORKS_DIR" ] && echo "ДА" || echo "НЕТ")"
  echo "   Права на чтение исходной: $([ -r "$PYTHON_FRAMEWORK_SRC" ] && echo "ДА" || echo "НЕТ")"
  echo "   Права на запись в целевую: $([ -w "$FRAMEWORKS_DIR" ] && echo "ДА" || echo "НЕТ")"
  exit 1
}
echo "✅ Python.framework СКОПИРОВАН"

# Проверяем результат копирования
COPIED_FRAMEWORK="$FRAMEWORKS_DIR/Python.framework"
if [ -d "$COPIED_FRAMEWORK" ]; then
  echo "✅ КОПИЯ Python.framework СОЗДАНА: $COPIED_FRAMEWORK"
  echo "📁 СТРУКТУРА СКОПИРОВАННОГО FRAMEWORK:"
  find "$COPIED_FRAMEWORK" -maxdepth 3 -type d | head -20
else
  echo "❌❌❌ КОПИЯ Python.framework НЕ СОЗДАНА!"
  exit 1
fi

# Создаём venv
echo "→ СОЗДАЁМ VIRTUALENV..."
PYTHON_BIN="$FRAMEWORKS_DIR/Python.framework/Versions/$PYTHON_VERSION/bin/python3"
echo "   Путь к Python binary: $PYTHON_BIN"

if [ ! -f "$PYTHON_BIN" ]; then
  echo "❌❌❌ Python бинарник НЕ НАЙДЕН в $PYTHON_BIN"
  echo "🔍 ПОИСК ДОСТУПНЫХ ВЕРСИЙ:"
  
  VERSIONS_DIR="$FRAMEWORKS_DIR/Python.framework/Versions"
  if [ -d "$VERSIONS_DIR" ]; then
    echo "📁 ДОСТУПНЫЕ ВЕРСИИ В $VERSIONS_DIR:"
    ls -la "$VERSIONS_DIR"
    
    for version_dir in "$VERSIONS_DIR"/*; do
      if [ -d "$version_dir" ]; then
        version_name=$(basename "$version_dir")
        python_path="$version_dir/bin/python3"
        echo "   Версия $version_name: $([ -f "$python_path" ] && echo "✅ python3 найден" || echo "❌ python3 НЕ найден")"
      fi
    done
  else
    echo "❌ ПАПКА VERSIONS НЕ СУЩЕСТВУЕТ: $VERSIONS_DIR"
  fi
  exit 1
else
  echo "✅ Python binary НАЙДЕН: $PYTHON_BIN"
  echo "📊 ИНФОРМАЦИЯ О Python:"
  "$PYTHON_BIN" --version 2>&1 || echo "Ошибка получения версии"
fi

echo "🔄 СОЗДАНИЕ VENV (это может занять время)..."
echo "   Команда: '$PYTHON_BIN' -m venv '$VENV_DST'"

"$PYTHON_BIN" -m venv "$VENV_DST" || {
  echo "❌❌❌ НЕ УДАЛОСЬ СОЗДАТЬ VENV"
  echo "🔍 ОТЛАДОЧНАЯ ИНФОРМАЦИЯ:"
  echo "   Python binary существует: $([ -f "$PYTHON_BIN" ] && echo "ДА" || echo "НЕТ")"
  echo "   Python binary исполняемый: $([ -x "$PYTHON_BIN" ] && echo "ДА" || echo "НЕТ")"
  echo "   Целевая папка для venv: $VENV_DST"
  echo "   Родительская папка существует: $([ -d "$(dirname "$VENV_DST")" ] && echo "ДА" || echo "НЕТ")"
  echo "   Права на запись в родительскую: $([ -w "$(dirname "$VENV_DST")" ] && echo "ДА" || echo "НЕТ")"
  exit 1
}

echo "✅ VENV СОЗДАН: $VENV_DST"
echo "📁 СТРУКТУРА СОЗДАННОГО VENV:"
find "$VENV_DST" -maxdepth 2 -type d | head -10

# Устанавливаем зависимости
echo "→ УСТАНАВЛИВАЕМ ЗАВИСИМОСТИ ИЗ requirements.txt..."
REQUIREMENTS_FILE="$PROJECT_DIR/Resources/requirements.txt"
echo "   Файл requirements: $REQUIREMENTS_FILE"

if [ ! -f "$REQUIREMENTS_FILE" ]; then
  echo "❌❌❌ ФАЙЛ requirements.txt НЕ НАЙДЕН: $REQUIREMENTS_FILE"
  echo "🔍 ПОИСК requirements.txt:"
  find "$PROJECT_DIR" -name "requirements.txt" -type f 2>/dev/null || echo "Файл не найден нигде в проекте"
  exit 1
else
  echo "✅ requirements.txt НАЙДЕН"
  echo "📋 СОДЕРЖИМОЕ requirements.txt:"
  cat "$REQUIREMENTS_FILE"
fi

echo "🔄 АКТИВАЦИЯ VENV И УСТАНОВКА ПАКЕТОВ..."
VENV_ACTIVATE="$VENV_DST/bin/activate"
VENV_PYTHON="$VENV_DST/bin/python3"
VENV_PIP="$VENV_DST/bin/pip"

echo "   activate script: $VENV_ACTIVATE"
echo "   venv python: $VENV_PYTHON"
echo "   venv pip: $VENV_PIP"

# Проверяем файлы venv
for venv_file in "$VENV_ACTIVATE" "$VENV_PYTHON" "$VENV_PIP"; do
  if [ -f "$venv_file" ]; then
    echo "   ✅ НАЙДЕН: $(basename "$venv_file")"
  else
    echo "   ❌ НЕ НАЙДЕН: $(basename "$venv_file") по пути $venv_file"
  fi
done

source "$VENV_ACTIVATE"
echo "✅ VENV АКТИВИРОВАН"

echo "📊 ИНФОРМАЦИЯ О АКТИВИРОВАННОМ ОКРУЖЕНИИ:"
which python3
which pip
python3 --version 2>&1 || echo "Ошибка получения версии Python"
pip --version 2>&1 || echo "Ошибка получения версии pip"

echo "🔄 УСТАНОВКА ПАКЕТОВ..."
pip install -r "$REQUIREMENTS_FILE" || {
  echo "❌❌❌ НЕ УДАЛОСЬ УСТАНОВИТЬ ЗАВИСИМОСТИ"
  echo "🔍 ОТЛАДОЧНАЯ ИНФОРМАЦИЯ:"
  echo "   pip доступен: $(which pip || echo "НЕТ")"
  echo "   requirements читаемый: $([ -r "$REQUIREMENTS_FILE" ] && echo "ДА" || echo "НЕТ")"
  exit 1
}

echo "✅ ЗАВИСИМОСТИ УСТАНОВЛЕНЫ"
echo "📦 УСТАНОВЛЕННЫЕ ПАКЕТЫ:"
pip list | head -20

deactivate
echo "✅ VENV ДЕАКТИВИРОВАН"

# Copy Python resources (scripts and models) to app bundle
echo "→ КОПИРУЕМ PYTHON РЕСУРСЫ (скрипты и модели)..."

# Copy autocomplete.py
AUTOCOMPLETE_SRC="$PROJECT_DIR/Resources/autocomplete.py"
echo "   Копируем autocomplete.py из: $AUTOCOMPLETE_SRC"
if [ -f "$AUTOCOMPLETE_SRC" ]; then
  cp "$AUTOCOMPLETE_SRC" "$RESOURCES_DIR/" || {
    echo "❌❌❌ НЕ УДАЛОСЬ СКОПИРОВАТЬ autocomplete.py"
    exit 1
  }
  echo "✅ autocomplete.py СКОПИРОВАН"
else
  echo "❌❌❌ autocomplete.py НЕ НАЙДЕН: $AUTOCOMPLETE_SRC"
  exit 1
fi

# Copy model file if it exists
MODEL_SRC="$PROJECT_DIR/Resources/minillm_export.pt"
echo "   Копируем minillm_export.pt из: $MODEL_SRC"
if [ -f "$MODEL_SRC" ]; then
  cp "$MODEL_SRC" "$RESOURCES_DIR/" || {
    echo "❌❌❌ НЕ УДАЛОСЬ СКОПИРОВАТЬ minillm_export.pt"
    exit 1
  }
  echo "✅ minillm_export.pt СКОПИРОВАН"
else
  echo "⚠️ minillm_export.pt НЕ НАЙДЕН в Resources, пропускаем: $MODEL_SRC"
  echo "🔍 ПОИСК ФАЙЛА МОДЕЛИ:"
  find "$PROJECT_DIR" -name "minillm_export.pt" -type f 2>/dev/null || echo "Файл модели не найден нигде в проекте"
fi

# Copy requirements.txt for reference
REQ_SRC="$PROJECT_DIR/Resources/requirements.txt"
echo "   Копируем requirements.txt из: $REQ_SRC"
if [ -f "$REQ_SRC" ]; then
  cp "$REQ_SRC" "$RESOURCES_DIR/" || {
    echo "❌❌❌ НЕ УДАЛОСЬ СКОПИРОВАТЬ requirements.txt"
    exit 1
  }
  echo "✅ requirements.txt СКОПИРОВАН"
else
  echo "❌❌❌ requirements.txt НЕ НАЙДЕН: $REQ_SRC"
fi

echo "📁 ФИНАЛЬНОЕ СОДЕРЖИМОЕ RESOURCES:"
ls -la "$RESOURCES_DIR" || echo "Ошибка чтения Resources"

# Патчим пути
echo "→ ПАТЧИМ ПУТИ В python3..."
PYBIN="$VENV_DST/bin/python3"
echo "   Python binary для патчинга: $PYBIN"

if [ ! -f "$PYBIN" ]; then
  echo "❌❌❌ Python binary для патчинга НЕ НАЙДЕН: $PYBIN"
  exit 1
fi

LIBPYTHON_OLD="/Library/Frameworks/Python.framework/Versions/$PYTHON_VERSION/lib/libpython${PYTHON_VERSION}.dylib"
LIBPYTHON_NEW="@executable_path/../../../Frameworks/Python.framework/Versions/$PYTHON_VERSION/lib/libpython${PYTHON_VERSION}.dylib"

echo "   Заменяем путь:"
echo "   ИЗ: $LIBPYTHON_OLD"
echo "   В:  $LIBPYTHON_NEW"

install_name_tool -change \
  "$LIBPYTHON_OLD" \
  "$LIBPYTHON_NEW" \
  "$PYBIN" || {
  echo "❌❌❌ НЕ УДАЛОСЬ ПАТЧИТЬ libpython"
  echo "🔍 ОТЛАДОЧНАЯ ИНФОРМАЦИЯ:"
  echo "   install_name_tool доступен: $(which install_name_tool || echo "НЕТ")"
  echo "   Python binary существует: $([ -f "$PYBIN" ] && echo "ДА" || echo "НЕТ")"
  otool -L "$PYBIN" 2>/dev/null || echo "Не удалось проанализировать зависимости"
  exit 1
}

echo "✅ ПУТИ В python3 ПРОПАТЧЕНЫ"

# Переподписываем python3
echo "→ ПЕРЕПОДПИСЫВАЕМ python3..."
codesign --force --sign - "$PYBIN" || {
  echo "❌❌❌ НЕ УДАЛОСЬ ПЕРЕПОДПИСАТЬ python3"
  echo "🔍 ОТЛАДОЧНАЯ ИНФОРМАЦИЯ:"
  echo "   codesign доступен: $(which codesign || echo "НЕТ")"
  exit 1
}

echo "✅ python3 ПЕРЕПОДПИСАН"

echo "🎉🎉🎉 УПАКОВКА .app ЗАВЕРШЕНА УСПЕШНО! 🎉🎉🎉"
echo "📊 ФИНАЛЬНАЯ СТАТИСТИКА:"
echo "   Размер приложения: $(du -sh "$APP_PATH" 2>/dev/null || echo "Не удалось определить")"
echo "   Время завершения: $(date)"

echo "🔍 ФИНАЛЬНАЯ ПРОВЕРКА СТРУКТУРЫ:"
echo "📁 СОДЕРЖИМОЕ APP:"
ls -la "$APP_PATH/Contents/" 2>/dev/null || echo "Ошибка чтения Contents"

echo "📁 СОДЕРЖИМОЕ RESOURCES:"
ls -la "$RESOURCES_DIR" 2>/dev/null || echo "Ошибка чтения Resources"

echo "📁 СОДЕРЖИМОЕ VENV:"
ls -la "$VENV_DST" 2>/dev/null || echo "Ошибка чтения venv"

echo "📁 СОДЕРЖИМОЕ VENV/BIN:"
ls -la "$VENV_DST/bin" 2>/dev/null || echo "Ошибка чтения venv/bin"

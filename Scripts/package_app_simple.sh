#!/bin/bash

# ===================================================================
# КОНФИГУРАЦИЯ - единственное место где задаются все пути и настройки
# ===================================================================

# Пути к приложению
APP_PATH="$BUILT_PRODUCTS_DIR/$PRODUCT_NAME.app"
RESOURCES_DIR="$APP_PATH/Contents/Resources"
VENV_DST="$RESOURCES_DIR/venv"

# Версии и имена
PYTHON_VERSION="3.12"  # По умолчанию, но будет автоопределена позже
VENV_NAME="venv"

# URL для скачивания готового Python.framework (встроенный в приложения)
PYTHON_FRAMEWORK_URL="https://github.com/astral-sh/python-build-standalone/releases/download/20250708/cpython-3.12.11+20250708-aarch64-apple-darwin-install_only.tar.gz"

# Проверка локального Python.framework
PYTHON_FRAMEWORK_LOCAL="/Library/Frameworks/Python.framework"

# Имя файла модели
MODEL_FILE_NAME="gemma-3-4b-pt-q4_0.gguf"

# ===================================================================
# ПРОВЕРКИ И ПОДГОТОВКА
# ===================================================================

echo "🔍 Проверяем приложение..."
if [ ! -d "$APP_PATH" ]; then
  echo "❌ Приложение не найдено: $APP_PATH"
  echo "💡 Сначала соберите приложение в Xcode"
  exit 1
fi
echo "✅ .app найден"

# Создаем папку Resources если её нет
mkdir -p "$RESOURCES_DIR"

# ===================================================================
# ПОЛУЧЕНИЕ PYTHON.FRAMEWORK
# ===================================================================

echo "📋 Получаем Python.framework..."

# Проверяем локальный Python.framework
if [ -d "$PYTHON_FRAMEWORK_LOCAL" ]; then
  echo "✅ Найден локальный Python.framework: $PYTHON_FRAMEWORK_LOCAL"
  cp -R "$PYTHON_FRAMEWORK_LOCAL" "$RESOURCES_DIR/"
  echo "✅ Локальный Python.framework скопирован"
else
  echo "⚠️  Локальный Python.framework не найден: $PYTHON_FRAMEWORK_LOCAL"
  echo "📥 Скачиваем готовый Python.framework для встраивания..."
  
  # Создаем временную папку для скачивания
  TEMP_DIR=$(mktemp -d)
  trap "rm -rf $TEMP_DIR" EXIT
  
  ARCHIVE_PATH="$TEMP_DIR/python-framework.tar.gz"
  
  echo "🌐 Скачиваем с: $PYTHON_FRAMEWORK_URL"
  if curl -L -o "$ARCHIVE_PATH" "$PYTHON_FRAMEWORK_URL"; then
    echo "✅ Python.framework скачан"
    
    echo "📦 Распаковываем архив..."
    cd "$TEMP_DIR"
    if tar -xzf "$ARCHIVE_PATH"; then
      echo "✅ Архив распакован"
      
      # Ищем Python.framework в распакованном архиве
      EXTRACTED_FRAMEWORK=$(find . -name "python" -type d | head -1)
      if [ -n "$EXTRACTED_FRAMEWORK" ]; then
        echo "📋 Копируем Python.framework..."
        
        # Копируем с разыменованием ссылок
        if cp -R -L "$EXTRACTED_FRAMEWORK" "$RESOURCES_DIR/Python.framework"; then
          echo "✅ Python.framework скопирован с разыменованием ссылок"
        else
          echo "⚠️  Не удалось скопировать с разыменованием, пробуем rsync..."
          if rsync -avL "$EXTRACTED_FRAMEWORK/" "$RESOURCES_DIR/Python.framework/"; then
            echo "✅ Python.framework скопирован через rsync"
          else
            echo "❌ Не удалось скопировать Python.framework"
            exit 1
          fi
        fi
      else
        echo "❌ Python.framework не найден в архиве"
        echo "💡 Проверьте содержимое: $(find . -type d | head -10)"
        exit 1
      fi
    else
      echo "❌ Не удалось распаковать архив"
      exit 1
    fi
  else
    echo "❌ Не удалось скачать Python.framework"
    echo "💡 Проверьте интернет-соединение"
    exit 1
  fi
fi

# ===================================================================
# АВТООПРЕДЕЛЕНИЕ ВЕРСИИ PYTHON
# ===================================================================

echo "📋 Определяем версию Python..."
ACTUAL_PYTHON_VERSION=$(ls "$RESOURCES_DIR/Python.framework/Versions/" | grep -E "^[0-9]+\.[0-9]+$" | head -1)
if [ -n "$ACTUAL_PYTHON_VERSION" ]; then
  PYTHON_VERSION="$ACTUAL_PYTHON_VERSION"
  echo "✅ Обнаружена версия Python: $PYTHON_VERSION"
else
  echo "⚠️  Не удалось автоопределить версию Python, используем: $PYTHON_VERSION"
fi

# ===================================================================
# СОЗДАНИЕ VENV
# ===================================================================

echo "📋 Создаем venv..."

# Используем локальный Python.framework для создания venv
LOCAL_PYTHON="$RESOURCES_DIR/Python.framework/Versions/$PYTHON_VERSION/bin/python3"

if [ ! -f "$LOCAL_PYTHON" ]; then
  echo "❌ Локальный Python не найден: $LOCAL_PYTHON"
  exit 1
fi

echo "✅ Найден локальный Python: $LOCAL_PYTHON"
echo "🔍 Размер: $(ls -lh "$LOCAL_PYTHON" 2>/dev/null | awk '{print $5}' || echo "НЕИЗВЕСТНО")"

# Удаляем старый venv
echo "🗑️  Удаляем старый venv..."
rm -rf "$VENV_DST"

# Создаем venv с локальным Python
echo "🚀 Создаем venv..."
cd "$RESOURCES_DIR/Python.framework/Versions/$PYTHON_VERSION/bin"
./python3 -m venv "$VENV_DST"

if [ -f "$VENV_DST/bin/python" ]; then
  echo "✅ venv создан успешно"
else
  echo "❌ Не удалось создать venv"
  echo "💡 Пропускаем создание venv и установку зависимостей"
  echo "💡 Можно создать venv позже вручную"
fi

# ===================================================================
# УСТАНОВКА ЗАВИСИМОСТЕЙ
# ===================================================================

echo "📋 Устанавливаем зависимости..."

# Проверяем, создан ли venv
if [ -f "$VENV_DST/bin/python" ]; then
  echo "✅ Venv найден, устанавливаем зависимости..."
  source "$VENV_DST/bin/activate"

  echo "🔍 Проверяем Python в venv..."
  echo "   Python: $(which python)"
  echo "   Версия: $(python --version 2>&1)"

  echo "💿 Устанавливаем пакеты (сначала из кэша, потом из сети)..."
  pip install --find-links="$PROJECT_DIR/Resources/wheels" -r "$PROJECT_DIR/Resources/requirements.txt"

  echo "⬆️  Обновляем pip до последней версии…"
  pip install --upgrade pip

  echo "⬆️  Обновляем пакеты до свежих версий…"
  pip install --upgrade --upgrade-strategy eager --find-links="$PROJECT_DIR/Resources/wheels" -r "$PROJECT_DIR/Resources/requirements.txt"

  deactivate
  echo "✅ Зависимости установлены"
else
  echo "⏭️  Venv не найден, пропускаем установку зависимостей"
  echo "💡 Зависимости можно установить позже вручную"
fi

# ===================================================================
# КОПИРОВАНИЕ РЕСУРСОВ
# ===================================================================

echo "🖼️  Копируем ресурсы..."

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
echo "   Python.framework: $(du -sh "$RESOURCES_DIR/Python.framework" 2>/dev/null || echo "Не удалось определить")"
if [ -d "$VENV_DST" ]; then
  echo "   venv: $(du -sh "$VENV_DST" 2>/dev/null || echo "Не удалось определить")"
else
  echo "   ⏭️  venv: не создан"
fi
echo "   Время завершения: $(date)" 
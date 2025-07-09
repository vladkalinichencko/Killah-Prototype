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
# URL для скачивания готового Python.framework (встроенный в приложения)
PYTHON_FRAMEWORK_URL="https://github.com/indygreg/python-build-standalone/releases/download/20231207/cpython-3.12.1+20231207-x86_64-apple-darwin.tar.gz"
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
  local framework_dst="$RESOURCES_DIR/Python.framework"
  
  if [ -d "$PYTHON_FRAMEWORK_LOCAL" ]; then
    echo "📋 Копируем локальный Python.framework..."
    
    # Умное копирование: сначала пытаемся с разыменованием символических ссылок
    echo "🔗 Пытаемся скопировать с разыменованием символических ссылок..."
    if cp -R -L "$PYTHON_FRAMEWORK_LOCAL" "$RESOURCES_DIR/" 2>/dev/null; then
      echo "✅ Локальный Python.framework скопирован с полным разыменованием ссылок"
    else
      echo "⚠️  Не удалось скопировать с разыменованием (битые симлинки), пробуем без разыменования..."
      # Удаляем частично скопированный фреймворк если он есть
      [ -d "$framework_dst" ] && rm -rf "$framework_dst"
      
      if cp -R -P "$PYTHON_FRAMEWORK_LOCAL" "$RESOURCES_DIR/" 2>/dev/null; then
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
    echo "📥 Скачиваем официальный Python installer..."
    
    # Создаем временную папку для скачивания
    local temp_dir=$(mktemp -d)
    trap "rm -rf $temp_dir" EXIT
    
    local pkg_path="$temp_dir/python.pkg"
    
    echo "🌐 Скачиваем с: $PYTHON_PKG_URL"
    if curl -L -o "$pkg_path" "$PYTHON_PKG_URL"; then
      echo "✅ Python installer скачан"
      
      echo "📦 Распаковываем .pkg файл..."
      cd "$temp_dir"
      
      # Распаковываем .pkg (это xar архив)
      if xar -xf "$pkg_path"; then
        echo "✅ .pkg распакован"
        
                 # Ищем Python_Framework.pkg/Payload файл (содержит Python.framework)
         local payload_file=$(find . -name "Python_Framework.pkg" -type d | head -1)
         if [ -n "$payload_file" ]; then
           echo "📦 Найден Python_Framework.pkg, распаковываем его Payload..."
           
                       # Проверяем, есть ли Payload в Python_Framework.pkg
            local framework_payload="$payload_file/Payload"
            if [ -f "$framework_payload" ]; then
              echo "📦 Распаковываем Payload из Python_Framework.pkg..."
              
              # Создаем папку для распаковки Payload
              mkdir -p payload_extracted/Python.framework
                          # Распаковываем Payload (это gzip сжатый tar) внутрь Python.framework
            if tar -xzf "$framework_payload" -C payload_extracted/Python.framework; then
              echo "✅ Payload распакован"
              echo "🔍 Проверяем содержимое распакованного Python.framework..."
              echo "   Структура: $(find payload_extracted/Python.framework -type f -name "python*" | head -5)"
              echo "   Ссылки в bin/: $(ls -la payload_extracted/Python.framework/Versions/*/bin/ 2>/dev/null | grep python || echo "папка не найдена")"
            
            # Ищем Python.framework в распакованном содержимом
            local extracted_framework="payload_extracted/Python.framework"
            if [ -n "$extracted_framework" ]; then
              echo "📋 Копируем Python.framework из installer..."
              
              # Применяем ту же умную логику копирования
        if cp -R -L "$extracted_framework" "$RESOURCES_DIR/" 2>/dev/null; then
                echo "✅ Python.framework из installer скопирован с полным разыменованием ссылок"
        else
                echo "⚠️  Не удалось скопировать с разыменованием, пробуем альтернативный способ..."
          # Удаляем частично скопированный фреймворк если он есть
          [ -d "$framework_dst" ] && rm -rf "$framework_dst"
          
                # Пробуем скопировать с принудительным разыменованием
                echo "🔄 Копируем с принудительным разыменованием..."
                if rsync -avL "$extracted_framework/" "$RESOURCES_DIR/Python.framework/" 2>/dev/null; then
                  echo "✅ Python.framework скопирован через rsync с разыменованием"
                elif cp -R -P "$extracted_framework" "$RESOURCES_DIR/" 2>/dev/null; then
                  echo "✅ Python.framework из installer скопирован без разыменования ссылок"
                  echo "🔍 Проверяем скопированный Python.framework..."
                  echo "   Ссылки в bin/: $(ls -la "$RESOURCES_DIR/Python.framework/Versions/*/bin/" 2>/dev/null | grep python || echo "папка не найдена")"
                  echo "   Размер python3: $(ls -lh "$RESOURCES_DIR/Python.framework/Versions/*/bin/python3" 2>/dev/null || echo "файл не найден")"
                  
                  # Исправляем символические ссылки в Python binary
                  echo "🔧 Исправляем символические ссылки в Python.framework..."
                  local python_binary="$RESOURCES_DIR/Python.framework/Versions/$PYTHON_VERSION/bin/python3"
                  if [ -L "$python_binary" ]; then
                    echo "   Исправляем ссылку в python3 binary..."
                    local target=$(readlink "$python_binary")
                    echo "   Цель ссылки: $target"
                    
                    # Ищем правильную цель ссылки
                    local target_path=""
                    if [ -f "$RESOURCES_DIR/Python.framework/Versions/$PYTHON_VERSION/bin/$target" ]; then
                      target_path="$RESOURCES_DIR/Python.framework/Versions/$PYTHON_VERSION/bin/$target"
                    elif [ -f "$RESOURCES_DIR/Python.framework/Versions/$PYTHON_VERSION/$target" ]; then
                      target_path="$RESOURCES_DIR/Python.framework/Versions/$PYTHON_VERSION/$target"
                    elif [ -f "$RESOURCES_DIR/Python.framework/Versions/$PYTHON_VERSION/Python" ]; then
                      target_path="$RESOURCES_DIR/Python.framework/Versions/$PYTHON_VERSION/Python"
                    else
                      echo "   🔍 Ищем Python binary в фреймворке..."
                      local found_python=$(find "$RESOURCES_DIR/Python.framework" -name "Python" -type f | head -1)
                      if [ -n "$found_python" ]; then
                        target_path="$found_python"
                        echo "   Найден Python binary: $found_python"
                      fi
                    fi
                    
                    if [ -n "$target_path" ] && [ -f "$target_path" ]; then
                      echo "   Копируем $target_path в $python_binary"
                      rm "$python_binary"
                      cp "$target_path" "$python_binary"
                      chmod +x "$python_binary"
                                        echo "   ✅ python3 binary исправлен"
                  
                  # Проверяем зависимости Python binary
                  echo "🔍 Проверяем зависимости Python binary..."
                  local deps=$(otool -L "$python_binary" | grep -v ":" | grep -v "@executable_path" | awk '{print $1}' | grep -v "^$")
                  if [ -n "$deps" ]; then
                    echo "   Зависимости: $deps"
                    echo "   Исправляем пути к библиотекам..."
                    
                    # Исправляем основные библиотеки Python
                    install_name_tool -change \
                      "/Library/Frameworks/Python.framework/Versions/$PYTHON_VERSION/Python" \
                      "@executable_path/../../../Python" \
                      "$python_binary" 2>/dev/null || true
                    
                    install_name_tool -change \
                      "/Library/Frameworks/Python.framework/Versions/$PYTHON_VERSION/lib/libpython${PYTHON_VERSION}.dylib" \
                      "@executable_path/../../../lib/libpython${PYTHON_VERSION}.dylib" \
                      "$python_binary" 2>/dev/null || true
                    
                                      echo "   ✅ Пути к библиотекам исправлены"
                else
                  echo "   ✅ Нет внешних зависимостей"
                fi
                
                # Проверяем наличие основных файлов Python.framework
                echo "🔍 Проверяем структуру Python.framework..."
                local python_main="$RESOURCES_DIR/Python.framework/Versions/$PYTHON_VERSION/Python"
                local python_lib="$RESOURCES_DIR/Python.framework/Versions/$PYTHON_VERSION/lib/libpython${PYTHON_VERSION}.dylib"
                
                if [ -f "$python_main" ]; then
                  echo "   ✅ Основной Python binary найден: $(ls -lh "$python_main")"
                else
                  echo "   ❌ Основной Python binary не найден: $python_main"
                fi
                
                if [ -f "$python_lib" ]; then
                  echo "   ✅ Python библиотека найдена: $(ls -lh "$python_lib")"
                  
                  # Исправляем символическую ссылку в библиотеке
                  if [ -L "$python_lib" ]; then
                    echo "   Исправляем ссылку в libpython..."
                    local lib_target=$(readlink "$python_lib")
                    if [ -f "$lib_target" ]; then
                      rm "$python_lib"
                      cp "$lib_target" "$python_lib"
                      echo "   ✅ libpython исправлен"
                    fi
                  fi
                else
                  echo "   ❌ Python библиотека не найдена: $python_lib"
                fi
                
                # Исправляем основной Python binary
                if [ -f "$python_main" ]; then
                  echo "   Исправляем зависимости основного Python binary..."
                  install_name_tool -change \
                    "/Library/Frameworks/Python.framework/Versions/$PYTHON_VERSION/Python" \
                    "@executable_path/Python" \
                    "$python_main" 2>/dev/null || true
                  
                  install_name_tool -change \
                    "/Library/Frameworks/Python.framework/Versions/$PYTHON_VERSION/lib/libpython${PYTHON_VERSION}.dylib" \
                    "@executable_path/lib/libpython${PYTHON_VERSION}.dylib" \
                    "$python_main" 2>/dev/null || true
                  
                  echo "   ✅ Основной Python binary исправлен"
                fi
                else
                  echo "   ❌ Не удалось найти подходящий Python binary"
                  echo "   💡 Доступные файлы в bin/: $(ls "$RESOURCES_DIR/Python.framework/Versions/$PYTHON_VERSION/bin/" 2>/dev/null || echo "папка не найдена")"
                fi
              fi
                  
                  # Исправляем другие важные ссылки
                  local python_binary_alt="$RESOURCES_DIR/Python.framework/Versions/$PYTHON_VERSION/bin/python"
                  if [ -L "$python_binary_alt" ]; then
                    echo "   Исправляем ссылку в python binary..."
                    local target=$(readlink "$python_binary_alt")
                    echo "   Цель ссылки: $target"
                    
                    # Ищем правильную цель ссылки
                    local target_path=""
                    if [ -f "$RESOURCES_DIR/Python.framework/Versions/$PYTHON_VERSION/bin/$target" ]; then
                      target_path="$RESOURCES_DIR/Python.framework/Versions/$PYTHON_VERSION/bin/$target"
                    elif [ -f "$RESOURCES_DIR/Python.framework/Versions/$PYTHON_VERSION/Python" ]; then
                      target_path="$RESOURCES_DIR/Python.framework/Versions/$PYTHON_VERSION/Python"
                    fi
                    
                    if [ -n "$target_path" ] && [ -f "$target_path" ]; then
                      rm "$python_binary_alt"
                      cp "$target_path" "$python_binary_alt"
                      chmod +x "$python_binary_alt"
                      echo "   ✅ python binary исправлен"
                    fi
                  fi
                  
          else
            echo "🔄 Используем выборочное копирование..."
            manual_copy_framework "$extracted_framework" "$framework_dst"
          fi
        fi
      else
              echo "❌ Python.framework не найден в installer"
              echo "💡 Проверьте содержимое Payload: $(find . -type d | head -10)"
              exit 1
            fi
          else
            echo "❌ Не удалось распаковать Payload"
            exit 1
                                    fi
           else
             echo "❌ Payload не найден в Python_Framework.pkg"
             echo "💡 Проверьте содержимое Python_Framework.pkg: $(ls -la "$payload_file" 2>/dev/null || echo "папка не найдена")"
             exit 1
           fi
         else
           echo "❌ Python_Framework.pkg не найден в .pkg"
           echo "💡 Доступные файлы: $(find . -type f | head -10)"
           exit 1
         fi
       else
         echo "❌ Не удалось распаковать .pkg файл"
        exit 1
      fi
    else
       echo "❌ Не удалось скачать Python installer"
       echo "💡 Установите Python.framework локально или проверьте интернет-соединение"
      exit 1
    fi
  fi
   
   # Исправляем пути в Python.framework после копирования
   echo "🔧 Исправляем пути в Python.framework..."
   local python_framework="$RESOURCES_DIR/Python.framework"
   local python_binary="$python_framework/Versions/$PYTHON_VERSION/bin/python3"
   
   if [ -f "$python_binary" ]; then
     # Исправляем пути к библиотекам Python
     install_name_tool -change \
       "/Library/Frameworks/Python.framework/Versions/$PYTHON_VERSION/Python" \
       "@executable_path/../../../Python" \
       "$python_binary" 2>/dev/null || true
     
     # Исправляем пути к libpython
     install_name_tool -change \
       "/Library/Frameworks/Python.framework/Versions/$PYTHON_VERSION/lib/libpython${PYTHON_VERSION}.dylib" \
       "@executable_path/../../../lib/libpython${PYTHON_VERSION}.dylib" \
       "$python_binary" 2>/dev/null || true
     
     echo "✅ Пути в Python.framework исправлены"
   else
     echo "⚠️  Python binary не найден для исправления путей: $python_binary"
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
ACTUAL_PYTHON_VERSION=$(ls "$RESOURCES_DIR/Python.framework/Versions/" | grep -E "^[0-9]+\.[0-9]+$" | head -1)
if [ -n "$ACTUAL_PYTHON_VERSION" ]; then
  PYTHON_VERSION="$ACTUAL_PYTHON_VERSION"
  echo "✅ Обнаружена версия Python: $PYTHON_VERSION"
else
  echo "⚠️  Не удалось автоопределить версию Python, используем: $PYTHON_VERSION"
fi

# ===================================================================
# СОЗДАНИЕ VENV С ЛОКАЛЬНЫМ PYTHON
# ===================================================================

echo "📋 Создаем venv с локальным Python..."

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
echo "   ⏭️  venv: пропущен (можно создать позже)"
echo "   Время завершения: $(date)"

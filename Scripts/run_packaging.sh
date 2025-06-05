#!/usr/bin/env bash

# Скрипт для ручного запуска упаковки приложения

echo "🔧 РУЧНОЙ ЗАПУСК УПАКОВКИ ПРИЛОЖЕНИЯ"
echo "======================================"

# Определяем пути
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

echo "📂 ПУТИ:"
echo "   Скрипт: $SCRIPT_DIR"
echo "   Проект: $PROJECT_DIR"

# Ищем последнюю сборку
echo "🔍 ПОИСК ПОСЛЕДНЕЙ СБОРКИ..."

# Обычные места для Xcode DerivedData
DERIVED_DATA_PATHS=(
    "$HOME/Library/Developer/Xcode/DerivedData"
    "/Users/*/Library/Developer/Xcode/DerivedData"
)

BUILD_PRODUCTS_DIR=""
APP_PATH=""

for derived_path in "${DERIVED_DATA_PATHS[@]}"; do
    if [ -d "$derived_path" ]; then
        echo "   Проверяем: $derived_path"
        
        # Ищем папки проекта
        find "$derived_path" -type d -name "*Killah*Prototype*" -maxdepth 1 2>/dev/null | while read project_dir; do
            echo "     Найдена папка проекта: $project_dir"
            
            # Ищем Build/Products
            build_products=$(find "$project_dir" -type d -path "*/Build/Products/Debug" 2>/dev/null | head -1)
            if [ -n "$build_products" ]; then
                echo "     ✅ Build/Products найден: $build_products"
                
                # Ищем .app
                app_file=$(find "$build_products" -name "*.app" -type d 2>/dev/null | head -1)
                if [ -n "$app_file" ]; then
                    echo "     ✅ .app найден: $app_file"
                    echo "$build_products" > /tmp/killah_build_dir
                    echo "$app_file" > /tmp/killah_app_path
                fi
            fi
        done
    fi
done

# Читаем найденные пути
if [ -f /tmp/killah_build_dir ]; then
    BUILD_PRODUCTS_DIR=$(cat /tmp/killah_build_dir)
    rm /tmp/killah_build_dir
fi

if [ -f /tmp/killah_app_path ]; then
    APP_PATH=$(cat /tmp/killah_app_path)
    rm /tmp/killah_app_path
fi

if [ -z "$BUILD_PRODUCTS_DIR" ] || [ -z "$APP_PATH" ]; then
    echo "❌ НЕ УДАЛОСЬ НАЙТИ СБОРКУ ПРИЛОЖЕНИЯ"
    echo "💡 Сначала соберите проект в Xcode (⌘+B)"
    echo "💡 Либо укажите путь вручную:"
    echo "   export BUILT_PRODUCTS_DIR='/path/to/Build/Products/Debug'"
    echo "   ./Scripts/package_app.sh"
    exit 1
fi

echo "✅ НАЙДЕНА СБОРКА:"
echo "   BUILD_PRODUCTS_DIR: $BUILD_PRODUCTS_DIR" 
echo "   APP: $APP_PATH"

# Экспортируем переменную и запускаем основной скрипт
export BUILT_PRODUCTS_DIR="$BUILD_PRODUCTS_DIR"

echo ""
echo "🚀 ЗАПУСКАЕМ УПАКОВКУ..."
echo "========================"

"$SCRIPT_DIR/package_app.sh"

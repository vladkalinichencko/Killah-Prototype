# Руководство по упаковке приложения Killah Prototype

## Обзор

Killah Prototype — это macOS приложение, которое объединяет Swift UI с Python backend для обработки текста и машинного обучения. Система упаковки создает полностью автономное приложение, которое может работать на любом Mac без необходимости установки Python или дополнительных зависимостей.

## Архитектура приложения

```text
Killah Prototype.app/
├── Contents/
│   ├── MacOS/
│   │   └── Killah Prototype           # Главный исполняемый файл Swift
│   ├── Frameworks/
│   │   └── Python.framework/          # Встроенный Python runtime
│   └── Resources/
│       ├── venv/                      # Виртуальное окружение Python
│       ├── autocomplete.py            # Python скрипт для автодополнения
│       ├── minillm_export.pt          # Модель PyTorch
│       └── requirements.txt           # Зависимости Python
```

## Как работает упаковка

### 1. Автоматическое получение Python.framework

Скрипт упаковки (`Scripts/package_app.sh`) сначала проверяет наличие Python.framework:

- **Если найден локально** (`/Library/Frameworks/Python.framework`): копирует его
- **Если НЕ найден**: автоматически скачивает предварительно собранный архив

### 2. Создание изолированного окружения

1. В приложение копируется полный Python.framework
2. Создается виртуальное окружение (venv) внутри Resources
3. Устанавливаются Python зависимости из `requirements.txt`
4. Копируются все необходимые Python файлы и модели

### 3. Исправление путей

Поскольку Python.framework теперь находится внутри приложения, а не в системе:

1. Исправляются пути к библиотекам в исполняемом файле Python
2. Используются относительные пути `@executable_path`
3. Файлы переподписываются для macOS

## Использование системы

### Для разработчиков

**Требования:**

- macOS с Xcode
- Bash (zsh)
- curl (для скачивания Python.framework при необходимости)

**Сборка:**

1. Откройте проект в Xcode
2. Выберите схему "Killah Prototype"
3. Product → Archive (или обычная сборка)
4. Скрипт упаковки запустится автоматически

**Что делает скрипт:**

```bash
# Минимальная и понятная логика
1. Проверяет существование .app
2. Получает Python.framework (локально или скачивает)
3. Создает venv и устанавливает зависимости
4. Копирует ресурсы
5. Исправляет пути и переподписывает
```

### Для пользователей

**Требования:**

- Никаких! Приложение полностью автономное

**Что включено в .app:**

- Полный Python 3.12 runtime
- Все необходимые Python библиотеки
- PyTorch модель для машинного обучения
- Все скрипты и ресурсы

## Swift код - простые и прямые пути

В `LLMEngine.swift` все пути теперь прямые и явные:

```swift
private func getResourcePath() -> String {
    return Bundle.main.resourcePath ?? ""
}

private func setupPaths() {
    let resourcePath = getResourcePath()
    self.pythonPath = resourcePath + "/venv/bin/python3"
    self.scriptPath = resourcePath + "/autocomplete.py"
    self.modelPath = resourcePath + "/minillm_export.pt"
}
```

**Никаких:**

- Сложных поисков путей
- Альтернативных вариантов
- Диагностики и fallback логики

## Конфигурация

В начале `package_app.sh` есть секция конфигурации:

```bash
# КОНФИГУРАЦИЯ
PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
APP_NAME="Killah Prototype.app"
PYTHON_VERSION="3.12"
VENV_NAME="venv"

# URL для скачивания Python.framework если нет локально
PYTHON_FRAMEWORK_URL="https://github.com/python/cpython-bin-deps/..."
PYTHON_FRAMEWORK_LOCAL="/Library/Frameworks/Python.framework"
```

## Устранение неполадок

### Если не удается скачать Python.framework

1. Проверьте интернет-соединение
2. Убедитесь, что URL актуален
3. Установите Python.framework локально:

   ```bash
   # Скачайте официальный Python installer с python.org
   # Или установите через Homebrew:
   brew install python@3.12
   ```

### Если приложение не запускается

1. Проверьте логи в Console.app
2. Убедитесь, что все файлы скопированы:

   ```bash
   ls -la "Killah Prototype.app/Contents/Resources/"
   ls -la "Killah Prototype.app/Contents/Frameworks/"
   ```

### Если Python код не работает

1. Проверьте, что venv создан правильно
2. Убедитесь в правильности путей в Swift коде
3. Проверьте, что все зависимости установлены

## Преимущества новой системы

1. **Минимальность**: Только необходимый код без избыточности
2. **Робастность**: Работает даже если Python не установлен
3. **Портативность**: Одинаково работает у всех разработчиков
4. **Автономность**: Пользователям не нужно устанавливать зависимости
5. **Простота**: Понятная логика без сложных алгоритмов поиска

## Размеры

Типичные размеры компонентов:

- Python.framework: ~50-80 MB
- venv с зависимостями: ~100-200 MB  
- Модель PyTorch: зависит от размера модели
- Итого приложение: ~200-400 MB

Это нормально для современного macOS приложения с машинным обучением.

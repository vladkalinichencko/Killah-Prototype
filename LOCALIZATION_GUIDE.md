# Руководство по локализации Killah Prototype

## Что уже сделано

1. ✅ Созданы файлы локализации:
   - `Killah Prototype/en.lproj/Localizable.strings` (английский)
   - `Killah Prototype/ru.lproj/Localizable.strings` (русский)

2. ✅ Обновлен код для использования локализованных строк:
   - Добавлено расширение `String.localized`
   - Все строки в UI заменены на локализованные версии

3. ✅ Обновлен Info.plist:
   - Добавлены ключи `CFBundleLocalizations` и `CFBundleDevelopmentRegion`

## Что нужно сделать в Xcode

### 1. Добавить файлы локализации в проект

1. Откройте проект в Xcode
2. В навигаторе проекта найдите папку "Killah Prototype"
3. Правой кнопкой мыши → "Add Files to 'Killah Prototype'"
4. Выберите папки `en.lproj` и `ru.lproj`
5. Убедитесь, что выбрана опция "Add to target: Killah Prototype"

### 2. Проверить настройки локализации

1. Выберите проект в навигаторе
2. Выберите target "Killah Prototype"
3. Перейдите на вкладку "Info"
4. В разделе "Localizations" должны быть:
   - English (Development Language)
   - Russian

### 3. Проверить Info.plist

Убедитесь, что в Info.plist есть:
```xml
<key>CFBundleLocalizations</key>
<array>
    <string>en</string>
    <string>ru</string>
</array>
<key>CFBundleDevelopmentRegion</key>
<string>en</string>
```

## Тестирование локализации

### В симуляторе:
1. Запустите приложение
2. В симуляторе: Device → Language → Russian
3. Перезапустите приложение

### На устройстве:
1. Настройки → Основные → Язык и регион → iPhone Language → Русский
2. Перезапустите приложение

## Структура локализации

### Английский (en.lproj/Localizable.strings)
- Все строки на английском языке
- Используется как базовый язык

### Русский (ru.lproj/Localizable.strings)
- Все строки переведены на русский язык
- Поддерживает форматирование с аргументами

## Добавление новых строк

1. Добавьте строку в английский файл: `"New String" = "New String";`
2. Добавьте перевод в русский файл: `"New String" = "Новая строка";`
3. В коде используйте: `"New String".localized`

## Форматирование с аргументами

```swift
// В файлах локализации:
"%d models need to be downloaded" = "%d models need to be downloaded";
"%d models need to be downloaded" = "%d моделей нужно загрузить";

// В коде:
String(format: "%d models need to be downloaded".localized, missing.count)
``` 
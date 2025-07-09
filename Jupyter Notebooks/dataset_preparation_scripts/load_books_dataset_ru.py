import os
import re
import json
import chardet
from pathlib import Path

# Папка с .txt файлами
input_folder = Path(
    "C:/Users/serma/killah_project/datasets/TextDatasets/BooksDatasets/russian_books")
output_file = Path(
    "C:/Users/serma/killah_project/datasets/TextDatasets/BooksDatasets/long_books_ru_cleaned.jsonl")

# Привязка авторов к названиям (можно расширить)
AUTHOR_MAP = {
    "pushkina": "Пушкин",
    "pushkin": "Пушкин",
    "dostoevsky": "Достоевский",
    "dostoevskiy": "Достоевский",
    "tolstoy": "Толстой",
    "voina_i_mir": "Толстой",
}

# Очистка текста по правилам


def clean_text(text: str, top_cut=0.05, bottom_cut=0.05) -> str:
    # Удаление начала и конца
    n = len(text)
    # text = text[int(n * top_cut): int(n * (1 - bottom_cut))]
    # Символы разрывов, табуляции
    text = text.replace('\n', ' ').replace('\r', ' ').replace('\t', ' ')
    # Двойные пробелы
    text = re.sub(r'\s{2,}', ' ', text)
    # Удаление типографики
    garbage = [
        "•", "◊", "◆", "□", "■", "▶", "◦", "▪", "►", "※", "❖",
        "™", "©", "®", "—", "–", "―", "·", "…", "<<", ">>",
        "†", "‡", "§", "¶", "~", "¤", "°", "º", "×", "₽", "№"
    ]
    for symbol in garbage:
        text = text.replace(symbol, "")
    # Удаление символов не из языка (но оставляем .,!? и кавычки)
    text = re.sub(r"[^\w\sа-яА-ЯёЁ.,!?\"'():;\\/-]", "", text)
    # Финальная нормализация
    text = re.sub(r'\s{2,}', ' ', text)
    return text.strip()


# Обработка всех файлов
with output_file.open("w", encoding="utf-8") as fout:
    for file_path in input_folder.glob("*.txt"):
        with file_path.open("rb") as f:
            raw_data = f.read()
            result = chardet.detect(raw_data)
            encoding = result['encoding']

        with file_path.open("r", encoding=encoding, errors='replace') as f:
            raw = f.read()

        cleaned = clean_text(raw)

        # Определение заголовка
        title = file_path.stem.replace("_", " ").title()

        # Определение автора
        filename = file_path.stem.lower()
        author = "Неизвестен"
        for key in AUTHOR_MAP:
            if key in filename:
                author = AUTHOR_MAP[key]
                break

        # Сохранение строки
        json.dump({
            "title": title,
            "author": author,
            "text": cleaned
        }, fout, ensure_ascii=False)
        fout.write("\n")
        print(f"✅ Обработано: {title} — {author}")

print(f"\n🎯 Готово! Сохранено книг: {len(list(input_folder.glob('*.txt')))}")

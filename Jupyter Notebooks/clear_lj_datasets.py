import json
import re
from pathlib import Path

# Путь к входному и выходному файлам
input_file = Path(
    "C:/Users/serma/killah_project/datasets/TextDatasets/LJDatasets/lj_posts.jsonl")
output_file = Path(
    "C:/Users/serma/killah_project/datasets/TextDatasets/LJDatasets/cleaned_lj_posts.jsonl")


def clean_text(text):
    # Удаляем символы разрыва страницы и переноса строк
    text = re.sub(r'\r|\n', ' ', text)
    # Удаляем двойные пробелы и табуляцию
    text = re.sub(r'\s+', ' ', text)
    # Удаляем типографские артефакты и специальные символы
    text = re.sub(r'[^\w\s]', '', text)
    return text.strip()


# Открываем входной и выходной файлы
with open(input_file, 'r', encoding='utf-8') as infile, \
        open(output_file, 'w', encoding='utf-8') as outfile:
    for line in infile:
        # Загружаем элемент из JSON
        item = json.loads(line)
        # Очищаем текст
        item['text'] = clean_text(item['text'])
        # Записываем очищенный элемент в выходной файл
        outfile.write(json.dumps(item, ensure_ascii=False) + '\n')

print(f"Текст очищен и записан в {output_file}.")

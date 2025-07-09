import json
import random
from pathlib import Path

# Пути к входному и выходному файлам
input_file = Path(
    "C:/Users/serma/killah_project/datasets/TextDatasets/LJDatasets/cleaned_lj_posts.jsonl")
output_file = Path(
    "C:/Users/serma/killah_project/datasets/TextDatasets/LJDatasets/ragged_lj_posts.jsonl")

# Параметры нарезки
min_len = 500  # Уменьшаем минимальную длину фрагмента
max_len = 5000  # Уменьшаем максимальную длину фрагмента


def generate_ragged_contexts(text, text_id):
    text_length = len(text)
    if text_length < min_len:
        return [{"id": text_id, "text": text}]

    # Список для хранения фрагментов
    fragments = []

    # Начинаем с начала текста
    start = 0

    # Генерируем фрагменты до тех пор, пока не достигнем конца текста
    while start < text_length:
        # Оставшаяся длина текста
        remaining_length = text_length - start

        # Если оставшаяся длина текста меньше минимальной длины фрагмента, добавляем оставшийся текст как последний фрагмент
        if remaining_length < min_len:
            fragments.append({"id": f"{text_id}_{start}",
                             "text": text[start:text_length]})
            break

        # Случайно выбираем длину фрагмента
        fragment_length = random.randint(
            min_len, min(max_len, remaining_length))

        # Извлекаем фрагмент
        fragment = text[start:start + fragment_length]

        # Добавляем фрагмент в список
        fragments.append({"id": f"{text_id}_{start}", "text": fragment})

        # Обновляем начальную точку
        start += fragment_length

    return fragments


# Открываем входной и выходной файлы
with open(input_file, 'r', encoding='utf-8') as infile, \
        open(output_file, 'w', encoding='utf-8') as outfile:
    for line in infile:
        # Загружаем элемент из JSON
        item = json.loads(line)
        text = item['text']
        text_id = item.get('id', random.randint(0, 1000000))

        # Генерируем рваные контексты
        ragged_contexts = generate_ragged_contexts(text, text_id)

        # Записываем каждый фрагмент в выходной файл
        for context in ragged_contexts:
            outfile.write(json.dumps(context, ensure_ascii=False) + '\n')

print(f"Рваные контексты созданы и записаны в {output_file}.")

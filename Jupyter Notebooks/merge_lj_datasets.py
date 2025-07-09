import json
from pathlib import Path

# Список пользователей
USERS = ["feruza",
         "wolfox",
         "chingizid",
         "izubr",
         "tanyant",
         "volha",
         "haez",
         "marussia",
         "mantrabox",
         "borisakunin",
         "dr_piliulkin",
         "dglu",
         "bagirov",
         "zorich",
         "sv_loginow",
         "exler",
         "divov",
         "vadim_panov",
         "mumi_mapa",
         "red_atomic_tank",
         "tema"]

# Путь к папке с файлами
folder_path = Path(
    "C:/Users/serma/killah_project/datasets/TextDatasets/LJDatasets")

# Имя выходного файла
output_file = folder_path / "lj_posts.jsonl"

# Открываем выходной файл для записи
with open(output_file, 'w', encoding='utf-8') as outfile:
    # Проходим по каждому пользователю в списке
    for user in USERS:
        # Формируем имя файла
        filename = f"{user}_lj_posts.jsonl"
        filepath = folder_path / filename

        # Проверяем, существует ли файл
        if filepath.exists():
            # Открываем и читаем файл
            with open(filepath, 'r', encoding='utf-8') as infile:
                for line in infile:
                    # Записываем каждую строку в выходной файл
                    outfile.write(line)
        else:
            print(f"Файл {filename} не найден.")

print(f"Все файлы объединены в {output_file}.")

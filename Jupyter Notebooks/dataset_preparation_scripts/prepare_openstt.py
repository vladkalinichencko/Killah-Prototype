import os
import json
from pathlib import Path
from utils import convert_to_wav
from tqdm import tqdm

# Пути
BASE_DIR = Path("C:/Users/serma/killah_project/datasets/OpenSTT/asr_public_stories_2")
PROCESSED_DIR = Path("C:/Users/serma/killah_project/datasets/OpenSTT/processed")
AUDIO_OUT_DIR = PROCESSED_DIR / "audio"
TRANSCRIPTS_PATH = PROCESSED_DIR / "transcripts.jsonl"

# Создание выходных папок
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
AUDIO_OUT_DIR.mkdir(parents=True, exist_ok=True)

# Список для jsonl
entries = []

# Рекурсивный обход всех файлов
for root, _, files in os.walk(BASE_DIR):
    for file in files:
        file_path = Path(root) / file

        if file_path.suffix == ".opus":
            wav_filename = file_path.stem + ".wav"
            wav_output_path = AUDIO_OUT_DIR / wav_filename

            # Проверка на наличие транскрипта
            txt_path = file_path.with_suffix(".txt")
            if not txt_path.exists():
                print(f"[!] WARNING: Нет транскрипта для {file_path}")
                continue

            # Чтение транскрипта
            with open(txt_path, "r", encoding="utf-8") as f:
                transcript = f.read().strip()

            # Конвертация .opus → .wav
            success = convert_to_wav(file_path, wav_output_path, sample_rate=16000)
            if not success:
                continue

            # Добавление записи
            entry = {
                "audio_filepath": str(wav_output_path.resolve()),
                "text": transcript,
                "language": "ru",
                "source": "openstt"
            }
            entries.append(entry)

# Запись transcripts.jsonl
with open(TRANSCRIPTS_PATH, "w", encoding="utf-8") as jsonl_file:
    for entry in entries:
        jsonl_file.write(json.dumps(entry, ensure_ascii=False) + "\n")

print(f"\n✅ Готово: {len(entries)} аудиофайлов конвертировано и описано.")
print(f"📄 transcripts.jsonl: {TRANSCRIPTS_PATH}")

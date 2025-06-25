import os
import json
import shutil
from pathlib import Path

# Пути
SOURCE_ROOT = Path("C:/Users/serma/killah_project/datasets/TESS")
DEST_ROOT = SOURCE_ROOT / "processed"
DEST_AUDIO_DIR = DEST_ROOT / "audio"
TRANSCRIPTS_PATH = DEST_ROOT / "transcripts.jsonl"

# Создаем директории
DEST_ROOT.mkdir(parents=True, exist_ok=True)
DEST_AUDIO_DIR.mkdir(parents=True, exist_ok=True)

# Очищаем предыдущий файл, если он существует
if TRANSCRIPTS_PATH.exists():
    TRANSCRIPTS_PATH.unlink()

# Открываем jsonl файл для записи
with open(TRANSCRIPTS_PATH, 'w', encoding='utf-8') as jsonl_file:
    # Обход всех подкаталогов, кроме "processed"
    for subdir in SOURCE_ROOT.iterdir():
        if subdir.is_dir() and subdir.name != "processed":
            for wav_file in subdir.glob("*.wav"):
                filename = wav_file.name  # Пример: OAF_back_angry.wav
                parts = filename.replace(".wav", "").split("_")

                if len(parts) < 3:
                    continue  # Пропускаем некорректные имена

                speaker = parts[0]
                word = parts[1]
                emotion = parts[2].lower()

                # Путь для нового расположения файла
                new_path = DEST_AUDIO_DIR / filename
                shutil.move(wav_file, new_path)  # Используй shutil.move, если хочешь перемещать

                # Запись в jsonl
                entry = {
                    "audio_filepath": str(new_path.resolve()),
                    "text": f"say the word {word}",
                    "emotion": emotion,
                    "language": "en",
                    "source": "tess"
                }

                jsonl_file.write(json.dumps(entry, ensure_ascii=False) + "\n")

print("✅ Обработка завершена. Все аудиофайлы собраны, transcripts.jsonl сформирован.")

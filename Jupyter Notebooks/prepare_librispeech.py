import json
import re
from pathlib import Path
from tqdm import tqdm
from utils import convert_to_wav

# Путь к исходному датасету
LIBRI_ROOT = Path("../datasets/LibriSpeech/LibriSpeech/train-clean-100")
OUT_DIR = Path("../datasets/LibriSpeech/LibriSpeech/processed")
AUDIO_OUT_DIR = OUT_DIR / "audio"

# Создаем директории, если они не существуют
OUT_DIR.mkdir(parents=True, exist_ok=True)
AUDIO_OUT_DIR.mkdir(parents=True, exist_ok=True)

TRANSCRIPTIONS = []

# Регулярное выражение для парсинга строки транскрипта
transcript_pattern = re.compile(r'^(\d+-\d+-\d+) (.+)$')

# Функция для чтения транскриптов из .trans.txt файла
def read_transcripts(trans_file):
    with open(trans_file, 'r', encoding='utf-8') as file:
        lines = file.readlines()

    transcripts = {}
    for line in lines:
        match = transcript_pattern.match(line.strip())
        if match:
            audio_id, text = match.groups()
            transcripts[audio_id] = text

    return transcripts

# Ищем все .flac файлы во вложенных папках
for audio_file in tqdm(LIBRI_ROOT.rglob("*.flac"), desc="Processing LibriSpeech"):
    # Получаем ID аудиофайла
    audio_id = audio_file.stem

    # Находим соответствующий .trans.txt файл
    trans_file = audio_file.parent / f"{audio_id.rsplit('-', 1)[0]}.trans.txt"

    if not trans_file.exists():
        print(f"Транскрипт не найден для {audio_file}")
        continue

    # Читаем транскрипты из .trans.txt файла
    transcripts = read_transcripts(trans_file)

    # Получаем транскрипт для текущего аудиофайла
    text = transcripts.get(audio_id)
    if not text:
        print(f"Транскрипт не найден для {audio_id}")
        continue

    new_audio_path = AUDIO_OUT_DIR / f"{audio_id}.wav"
    if convert_to_wav(audio_file, new_audio_path):
        TRANSCRIPTIONS.append({
            "audio_filepath": str(new_audio_path.resolve()),
            "text": text,
            "language": "en",
            "source": "librispeech"
        })
    else:
        print(f"Ошибка конвертации {audio_file} в WAV формат")

# Сохраняем результат
jsonl_path = OUT_DIR / "transcripts.jsonl"
with open(jsonl_path, "w", encoding="utf-8") as out_f:
    for entry in TRANSCRIPTIONS:
        json.dump(entry, out_f, ensure_ascii=False)
        out_f.write("\n")

print(f"\n✅ Готово! Обработано {len(TRANSCRIPTIONS)} записей.")
print(f"→ JSONL: {jsonl_path}")

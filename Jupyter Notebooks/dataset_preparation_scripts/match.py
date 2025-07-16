import json
import sys
from pathlib import Path

def find_transcript(jsonl_path, wav_filename):
    # Читаем JSONL файл и ищем транскрипт для указанного WAV файла
    with open(jsonl_path, 'r', encoding='utf-8') as file:
        for line in file:
            entry = json.loads(line)
            if entry['audio_filepath'].endswith(wav_filename):
                return entry['text']
    return None


def match(wav_filename: str, dataset_name: str):
    jsonl_path = Path(f"C:/Users/serma/killah_project/datasets/{dataset_name}/processed/transcripts.jsonl")

    transcript = find_transcript(jsonl_path, wav_filename)

    if transcript:
        print(f"Транскрипт для {wav_filename}: {transcript}")
    else:
        print(f"Транскрипт для {wav_filename} не найден.")


if __name__ == "__main__":
    match("OAF_boat_sad.wav", "TESS")

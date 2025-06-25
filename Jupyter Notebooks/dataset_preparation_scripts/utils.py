import subprocess
from pathlib import Path

def convert_to_wav(input_path: Path, output_path: Path, sample_rate=16000):
    try:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        command = [
            "ffmpeg",
            "-y",  # перезаписывать без запроса
            "-i", str(input_path),
            "-ar", str(sample_rate),
            "-ac", "1",  # моно
            str(output_path)
        ]
        result = subprocess.run(command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return result.returncode == 0
    except Exception as e:
        print(f"FFmpeg error on {input_path}: {e}")
        return False

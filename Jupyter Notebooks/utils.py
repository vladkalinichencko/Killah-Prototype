from pydub import AudioSegment
from pathlib import Path

def convert_to_wav(input_path: Path, output_path: Path, sample_rate=16000):
    try:
        audio = AudioSegment.from_file(input_path)
        audio = audio.set_frame_rate(sample_rate).set_channels(1)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        audio.export(output_path, format="wav")
        return True
    except Exception as e:
        print(f"Conversion error: {input_path} -> {e}")
        return False
      

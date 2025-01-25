import os
from typing import Any, Optional, TextIO
from pyannote.audio import Pipeline, Audio
import whisper
from whisper.utils import WriteSRT, WriteVTT
from whisper import Whisper
import torch
import json
from math import ceil, floor
from pydub import AudioSegment

# 设置音频文件路径
AUDIO_FILE = "/content/drive/MyDrive/Whisper_6method/v058_MECOLAB_minus3_mins.wav"
OUTPUT_FOLDER = "/content/drive/MyDrive/Whisper_6method/Method5/"
HF_AUTH_TOKEN = "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"

class AppendResultsMixin:
    first_call: bool = True
    output_path: str = ''
    
    def get_path_and_open_mode(self, *, audio_path: str, dir: str, ext: str) -> tuple[str, str]:
        mode: str
        if self.first_call:
            audio_basename = os.path.basename(audio_path)
            audio_basename = os.path.splitext(audio_basename)[0]
            self.output_path: str = os.path.join(dir, audio_basename + "." + ext)
            self.first_call = False
            mode = 'w'
        else:
            mode = 'a'
        return self.output_path, mode

class WriteJSONFormat(AppendResultsMixin):
    def __init__(self, output_dir: str):
        self.output_dir = output_dir
        self.data = []
        super().__init__()

    def __call__(self, result: dict, audio_path: str, speaker: str,
                 options: Optional[dict] = None, **kwargs):
        # 遍历段落
        print(result.keys())  # 调试打印
        if "language" in result:
            print(f"Language found: {result['language']}")
        else:
            print("No language key found")

        for segment in result["segments"]:
            # 构造 words 列表
            words_data = []
            if "words" in segment:
                for w in segment["words"]:
                    # Whisper 默认并不包含 'probability'
                    # 如果你的版本带了 probability，就存
                    # 若没这个字段，你可以 w.get("probability", 0.0)
                    words_data.append({
                        "word": w["word"],
                        "start": w["start"],
                        "end": w["end"],
                        "probability": w.get("probability", 0.0)
                    })

            # 拼出 Timestamp 字符串: "0.03 - 2.71"
            start_time = segment["start"]
            end_time = segment["end"]
            timestamp_str = f"{start_time:.2f} - {end_time:.2f}"
            
            # 将想要的字段写入 self.data
            self.data.append({
                "Timestamp": timestamp_str,
                "Speaker": speaker,
                "Words": words_data,
                "Language": result['language']
            })

    def write_to_file(self, audio_path: str):
        output_filename = "Method5_medium_WBW.json"
        path = os.path.join(self.output_dir, output_filename)
        print(f"Total entries before writing: {len(self.data)}")
        print("First entry sample:", self.data[0])
        
        with open(path, 'w', encoding="utf-8") as f:
            json.dump(self.data, f, ensure_ascii=False, indent=2)
            
        # 验证写入后的文件
        with open(path, 'r', encoding="utf-8") as f:
            written_data = json.load(f)
        print(f"Total entries in written file: {len(written_data)}")

def diarize_audio(HF_AUTH_TOKEN, AUDIO_FILE):
    pipeline = Pipeline.from_pretrained(
    "pyannote/speaker-diarization-3.1",
    use_auth_token=HF_AUTH_TOKEN)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    pipeline.to(torch.device(device))
    print(f"Diarize audio on {device}")
    io = Audio(mono='downmix', sample_rate=16000)
    waveform, sample_rate = io(AUDIO_FILE)
    diarization = pipeline({"waveform": waveform, "sample_rate": sample_rate})
    return diarization

class WriteSRTIncremental(AppendResultsMixin, WriteSRT):
    srt_index: int = 1

    def __call__(self, result: dict, audio_path: str, options: Optional[dict] = None, **kwargs):
        path, mode = self.get_path_and_open_mode(audio_path=audio_path, dir=self.output_dir, ext=self.extension)
        with open(path, mode, encoding="utf-8") as f:
            self.write_result(result, file=f, options=options, **kwargs)
            
    def write_result(self, result: dict, file: TextIO, options: Optional[dict] = None, **kwargs):
        for (start, end, text) in self.iterate_result(result, options, **kwargs):
            print(f"{self.srt_index}\n{start} --> {end}\n{text}\n", file=file, flush=True)
            self.srt_index += 1

class WriteVTTIncremental(AppendResultsMixin, WriteVTT):
    def __call__(self, result: dict, audio_path: str, options: Optional[dict] = None, **kwargs):
        path, mode = self.get_path_and_open_mode(audio_path=audio_path, dir=self.output_dir, ext=self.extension)
        with open(path, mode, encoding="utf-8") as f:
            if mode != 'a': 
                print("WEBVTT\n", file=f)
            self.write_result(result, file=f, options=options, **kwargs)

    def write_result(self, result: dict, file: TextIO, options: Optional[dict] = None, **kwargs):
        for start, end, text in self.iterate_result(result, options, **kwargs):
            print(f"{start} --> {end}\n{text}\n", file=file, flush=True)

class WriteCustomFormat(AppendResultsMixin):
    def __init__(self, output_dir: str):
        self.output_dir = output_dir
        super().__init__()

    def __call__(self, result: dict, audio_path: str, speaker: str, options: Optional[dict] = None, **kwargs):
        path, mode = self.get_path_and_open_mode(audio_path=audio_path, dir=self.output_dir, ext="txt")
        with open(path, mode, encoding="utf-8") as f:
            self.write_result(result, file=f, speaker=speaker, options=options, **kwargs)

    def write_result(self, result: dict, file: TextIO, speaker: str, options: Optional[dict] = None, **kwargs):
        for segment in result['segments']:
            timestamp = f"{self.format_timestamp(segment['start'])} - {self.format_timestamp(segment['end'])}"
            text = segment['text'].strip()
            language = result['language']
            print(f"{timestamp} | {speaker} | {text} | {language}", file=file, flush=True)

    def format_timestamp(self, seconds: float) -> str:
        minutes = int(seconds // 60)
        seconds = int(seconds % 60)
        return f"{minutes:02d}:{seconds:02d}"

def map_speaker_label(speaker):
    speaker_mapping = {
        'SPEAKER_00': 'Speaker 1 System',
        'SPEAKER_01': 'Speaker 2 Child',
        'SPEAKER_02': 'Speaker 3 Parent',
        'SPEAKER_03': 'Speaker 4 Experimenter',
        'SPEAKER_04': 'Speaker 5 Multiple Speakers',
        'SPEAKER_05': 'Speaker 6 Others',
    }
    return speaker_mapping.get(speaker, speaker)

class WhisperFacade:
    wmodel: Whisper

    def __init__(self, model:str, *, quantize=False) -> None:
        print("Initialize whisper")
        whisper_model = whisper.load_model(model)
        if quantize:
            print("Quantize")
            DTYPE = torch.qint8
            qmodel: Whisper = torch.quantization.quantize_dynamic(
                        whisper_model, {torch.nn.Linear}, dtype=DTYPE)
            del whisper_model
            self.wmodel = qmodel
        else:
            self.wmodel = whisper_model

    def _set_timing_for(self, segment: dict[str, float], offset: float) -> None:
        s = segment
        s['start'] += offset 
        s['end']   += offset
        if 'words' in s:
            for w in s['words']:
                w['start'] += offset
                w['end'] += offset

    def load_audio(self, file_path: str):
        self.audio = whisper.load_audio(file_path)
    
    def transcribe(self, *, start: float, end: float, options: dict[str, Any] ) -> dict[str, Any]:
        SAMPLE_RATE = 16_000
        start_index = floor(start * SAMPLE_RATE)
        end_index = ceil(end * SAMPLE_RATE)
        audio_segment = self.audio[start_index:end_index]
        result = whisper.transcribe(self.wmodel, audio_segment, **options)
        segments = result['segments']
        for s in segments:
            self._set_timing_for(segment=s, offset=start)
        return result

def main(use_quantization=False):
    torch.set_num_threads(6)  # 根据Colab的CPU调整线程数

    print("Processing audio...")
    model = WhisperFacade(model='medium', quantize=use_quantization)

    srt_writer = WriteSRTIncremental('/content') 
    custom_writer = WriteCustomFormat('/content') 
    json_writer = WriteJSONFormat(OUTPUT_FOLDER)  # 使用 OUTPUT_FOLDER

    whisper_options = {
        "verbose": None,
        "word_timestamps": True,
        "task": "transcribe",
        "suppress_tokens": "",
        "language": None,  # 自动检测语言
        "temperature": 0.0,  # 降低随机性
        "condition_on_previous_text": True  # 考虑上下文
    }

    writer_options = {"max_line_width": 55, "max_line_count": 2, "highlight_words": False}

    diarization = diarize_audio(HF_AUTH_TOKEN, AUDIO_FILE)
    model.load_audio(AUDIO_FILE)

    for turn, _, speaker in diarization.itertracks(yield_label=True):
        if turn.end - turn.start < 0.5:  # 忽略很短的语音片段
            continue
        mapped_speaker = map_speaker_label(speaker)
        result = model.transcribe(start=turn.start, end=turn.end, options=whisper_options)
        language = result['language']
        print(f"{mapped_speaker}: {turn.start:.1f}s - {turn.end:.1f}s, Language: {language}")
        
        srt_writer(result, AUDIO_FILE, writer_options)
        custom_writer(result, AUDIO_FILE, speaker=mapped_speaker)
        json_writer(result, AUDIO_FILE, speaker=mapped_speaker)

    # 在循环结束后，写入 JSON 文件
    json_writer.write_to_file(AUDIO_FILE)

    print("Processing complete. Custom format and JSON files generated.")

if __name__ == '__main__':
    use_quantization = False  # 在这里设置是否使用量化模型
    main(use_quantization)
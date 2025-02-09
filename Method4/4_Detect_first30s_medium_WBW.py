import whisper
from pydub import AudioSegment
import os
import json
import math

# 使用提供的路径
AUDIO_FILE = "{input_file}"
OUTPUT_FOLDER = "{output_folder}"

def split_audio(file_path, max_size_mb=25):
    audio = AudioSegment.from_wav(file_path)
    max_size_bytes = max_size_mb * 1024 * 1024
    duration_ms = len(audio)

    total_bytes = audio.frame_count() * audio.frame_width
    num_segments = max(1, math.ceil(total_bytes / max_size_bytes))

    segment_duration = duration_ms // num_segments

    segments = []
    for i in range(num_segments):
        start = i * segment_duration
        end = (i + 1) * segment_duration
        end = min(end, duration_ms)

        segment = audio[start:end]
        segment_path = os.path.join(OUTPUT_FOLDER, f"part_{i+1}.wav")
        segment.export(segment_path, format="wav")

        # 注意：这里返回的是秒数
        segments.append((segment_path, start / 1000, end / 1000))
    return segments, audio


def detect_language(file_path):
    """
    截取音频前30秒，用 Whisper tiny 模型快速转录并返回检测到的语言
    """
    audio = AudioSegment.from_wav(file_path)
    first_30_seconds = audio[:30000]  # 30000毫秒 = 30秒
    temp_file = os.path.join(OUTPUT_FOLDER, "temp_30s.wav")
    first_30_seconds.export(temp_file, format="wav")

    detected_language = None

    try:
        model = whisper.load_model("tiny") 
        result = model.transcribe(temp_file, language=None, word_timestamps=False)
        if "language" in result:
            detected_language = result["language"]
    except Exception as e:
        print(f"An error occurred during language detection: {e}")
    finally:
        if os.path.exists(temp_file):
            os.remove(temp_file)

    print(f"[Language Detection] => {detected_language}")
    return detected_language


def transcribe_audio_with_timestamps(file_path, language_param, model):
    """
    用指定模型对单个音频片段转录，返回 (segments_list, final_language_str)
    """
    try:
        options = {
            "verbose": None,
            "word_timestamps": True,
            "task": "transcribe",
            "suppress_tokens": ""
        }
        if language_param is not None:
            options["language"] = language_param

        result = model.transcribe(file_path, **options)
        
        final_language = result.get("language", None)
        segments = result["segments"] if "segments" in result else None
        
        return (segments, final_language)
    except Exception as e:
        print(f"An error occurred during transcription for {file_path}: {e}")
        return (None, None)


def process_transcription(segments, offset_ms):
    """
    为每个转录段添加全局偏移：
    offset_ms: 当前分段在原始音频中的起始时间（毫秒）
    将分段内的局部时间转换为全局时间（单位：秒）
    """
    processed_segments = []
    base_offset_sec = offset_ms / 1000.0
    for seg in segments:
        seg_copy = seg.copy()
        # 将局部时间加上偏移量
        seg_copy["start"] = base_offset_sec + seg["start"]
        seg_copy["end"] = base_offset_sec + seg["end"]
        # 处理词级时间戳
        if "words" in seg_copy:
            for w in seg_copy["words"]:
                w["start"] = base_offset_sec + w["start"]
                w["end"] = base_offset_sec + w["end"]
        processed_segments.append(seg_copy)
    return processed_segments


def format_timestamp(seconds: float) -> str:
    minutes = int(seconds // 60)
    secs = int(seconds % 60)
    return f"{minutes:02d}:{secs:02d}"


def format_segments(segments, final_language):
    """
    格式化每个转录段，直接使用已经调整过的全局时间
    """
    if not segments:
        return []
    formatted = []
    for seg in segments:
        start_time = seg["start"]
        end_time = seg["end"]
        segment_data = {
            "Timestamp": f"{start_time:.2f} - {end_time:.2f}",
            "text": seg["text"],
            "Words": [],
            "Language": final_language if final_language else "und"
        }
        if "words" in seg and seg["words"]:
            for w in seg["words"]:
                word_data = {
                    "word": w["word"],
                    "start": format_timestamp(w["start"]),
                    "end": format_timestamp(w["end"])
                }
                if "probability" in w:
                    word_data["probability"] = w["probability"]
                segment_data["Words"].append(word_data)
        formatted.append(segment_data)
    return formatted


def transcribe_audio_segments(segments_info, language_param, model):
    """
    对分割后的音频片段逐个调用转录，将局部时间转换为全局时间，
    并合并所有转录结果返回，同时记录最终使用的语言
    """
    all_transcriptions = []
    final_language_used = None

    # 遍历时解包 (segment_path, start, end)
    for i, (segment_path, start, end) in enumerate(segments_info, 1):
        print(f"Transcribing segment {i}/{len(segments_info)} => {segment_path}...")
        seg_result, seg_lang = transcribe_audio_with_timestamps(segment_path, language_param, model)
        if seg_result:
            ### CHANGED: 此处要传入毫秒, 因为 process_transcription(offset_ms=...) 里是 offset_ms/1000.0
            processed_segments = process_transcription(seg_result, start * 1000)
            all_transcriptions.extend(processed_segments)
            if final_language_used is None and seg_lang is not None:
                final_language_used = seg_lang
        else:
            print(f"Transcription failed for segment {i}")
    return (all_transcriptions, final_language_used)


def main():
    print("Detecting language from the first 30 seconds...")
    detected_language = detect_language(AUDIO_FILE)
    print(f"Detected language: {detected_language}")

    print("Splitting audio into segments...")
    ### CHANGED: 接收split_audio返回的两个值: (segments_info, audio)
    segments_info, audio = split_audio(AUDIO_FILE)
    print(f"Audio split into {len(segments_info)} segments.")

    print("Loading whisper 'medium' model for detailed transcription...")
    final_model = whisper.load_model("medium")  # 这里使用 small 模型

    all_segments, final_language = transcribe_audio_segments(segments_info, detected_language, final_model)
    if final_language is None:
        final_language = "und"

    formatted_segments = format_segments(all_segments, final_language)

    transcription_file = os.path.join(OUTPUT_FOLDER, "Method4_medium_WBW.json")
    with open(transcription_file, 'w', encoding='utf-8') as f:
        json.dump(formatted_segments, f, ensure_ascii=False, indent=2)
    print(f"Transcription saved to {transcription_file}")

    print("Cleaning up temporary files...")
    for seg_info in segments_info:
        if os.path.exists(seg_info[0]):
            os.remove(seg_info[0])
    print("Cleanup complete.")


if __name__ == "__main__":
    main()
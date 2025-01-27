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
    
    # 估计切分段数
    file_size_bytes = os.path.getsize(file_path)
    num_segments = math.ceil(file_size_bytes / max_size_bytes)
    if num_segments < 1:
        num_segments = 1

    segment_duration = duration_ms // num_segments

    segments = []
    for i in range(num_segments):
        start = i * segment_duration
        end = (i + 1) * segment_duration
        if end > duration_ms:
            end = duration_ms
        segment = audio[start:end]
        segment_path = os.path.join(OUTPUT_FOLDER, f"audio_part{i+1}.wav")
        segment.export(segment_path, format="wav")
        segments.append(segment_path)

    return segments

def detect_language(file_path):
    """
    1) 截取音频前30秒
    2) 用 whisper 本地小模型做 quick transcribe
    3) 如果成功，则返回 result["language"]
       如果失败，或者 result["language"] 不存在，就返回 None
    """
    audio = AudioSegment.from_wav(file_path)
    first_30_seconds = audio[:30000]  # 30000毫秒 = 30秒
    temp_file = os.path.join(OUTPUT_FOLDER, "temp_30s.wav")
    first_30_seconds.export(temp_file, format="wav")

    detected_language = None  # 改成 None，当检测失败时自动使用 None 进行后续识别

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

    # 如果检测失败或者没拿到 language，就返回 None
    print(f"[Language Detection] => {detected_language}")
    return detected_language

def transcribe_audio_with_timestamps(file_path, language_param, model):
    """
    使用本地模型 model.transcribe(...) 完成词级别转录。
    language_param 可以是具体语言字符串("en"/"zh")或 None(自动检测)。
    
    在这里做少量修改以获取最后大模型实际使用/检测到的语言。
    返回 (segments_list, final_language_str)
    """
    try:
        options = {
            "word_timestamps": True,
            "task": "transcribe",
            "verbose": None,
            "temperature": 0.0,
            "verbose": None,
            "suppress_tokens": "",
            "condition_on_previous_text": True  # 考虑上下文
        }
        if language_param is not None:
            options["language"] = language_param

        result = model.transcribe(file_path, **options)
        
        # 这行是新增的：从大模型的返回里获取最终语言
        final_language = result.get("language", None)
        
        # 取 segments 供后续拼接
        segments = result["segments"] if "segments" in result else None
        
        # 返回 (segments, final_language)
        return (segments, final_language)
    except Exception as e:
        print(f"An error occurred during transcription for {file_path}: {e}")
        return (None, None)

def format_timestamp(seconds):
    minutes = int(seconds // 60)
    sec = int(seconds % 60)
    return f"{minutes:02d}:{sec:02d}"

def format_segments(segments, final_language):
    """
    这里在原 format_segments 基础上，只加了一个 final_language 参数，
    用于把大模型最终使用的语言写进每段的 JSON。
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
            # 新增字段：记录大模型最终使用/检测到的语言
            "Language": final_language if final_language else "und"
        }
        if "words" in seg and seg["words"]:
            # 提取词级信息
            for w in seg["words"]:
                word_data = {
                    "word": w["word"],
                    "start": w["start"],
                    "end": w["end"]
                }
                # 如果包含 probability，可加上
                if "probability" in w:
                    word_data["probability"] = w["probability"]
                segment_data["Words"].append(word_data)
        formatted.append(segment_data)
    return formatted

def transcribe_audio_segments(segments_paths, language_param, model):
    """
    对分割后的音频片段逐个调用本地 whisper，并把结果合并。
    language_param 可为 None 或 "en" / "zh" 等。

    这里也做最小修改：返回 (all_transcriptions, final_language_used)
    方便后续输出 JSON 时知道究竟使用了哪种语言。
    """
    all_transcriptions = []
    final_language_used = None  # 记录大模型真实检测到的语言

    for i, seg_path in enumerate(segments_paths, 1):
        print(f"Transcribing segment {i}/{len(segments_paths)} => {seg_path}...")
        seg_result, seg_lang = transcribe_audio_with_timestamps(seg_path, language_param, model)
        if seg_result:
            all_transcriptions.extend(seg_result)
            # 如果之前 final_language_used 还没有值，则用第一个非空 seg_lang
            if final_language_used is None and seg_lang is not None:
                final_language_used = seg_lang
        else:
            print(f"Transcription failed for segment {i}")
    return (all_transcriptions, final_language_used)

def main():
    print("Detecting language from the first 30 seconds...")
    detected_language = detect_language(AUDIO_FILE)
    # 如果检测失败, detected_language 会是 None
    print(f"Detected language: {detected_language}")

    print("Splitting audio into segments...")
    audio_segments = split_audio(AUDIO_FILE)
    print(f"Audio split into {len(audio_segments)} segments.")

    # 加载大一点的模型进行详细转录 (可选: small / medium / large)
    print("Loading whisper 'medium' model for detailed transcription...")
    final_model = whisper.load_model("small")

    # 这里把 language 传给 transcribe_audio_segments, 如果是 None, 就自动检测
    all_segments, final_language = transcribe_audio_segments(audio_segments, detected_language, final_model)

    # 如果最终语言没检测到，就标记为 "und"
    if final_language is None:
        final_language = "und"

    # 调用 format_segments 时，把最后的语言也传进去
    formatted_segments = format_segments(all_segments, final_language)

    # 输出到 JSON
    transcription_file = os.path.join(OUTPUT_FOLDER, "Method4_small_WBW.json")
    with open(transcription_file, 'w', encoding='utf-8') as f:
        json.dump(formatted_segments, f, ensure_ascii=False, indent=2)
    print(f"Transcription saved to {transcription_file}")

    # 清理临时文件
    print("Cleaning up temporary files...")
    for seg_path in audio_segments:
        os.remove(seg_path)
    print("Cleanup complete.")

if __name__ == "__main__":
    main()
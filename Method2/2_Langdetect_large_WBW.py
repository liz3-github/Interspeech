import os
import math
import re
from pydub import AudioSegment
from langdetect import detect
import whisper
import json

# ---------- 全局设置 ----------
AUDIO_FILE = "{input_file}"
OUTPUT_FOLDER = "{output_folder}"
MAX_SEGMENT_MB = 25  

os.makedirs(OUTPUT_FOLDER, exist_ok=True)


# ========== 第一步：切分音频（示例：按大小分割） ==========
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

        segments.append((segment_path, start / 1000, end / 1000))
    return segments, audio


# ========== 辅助函数：如果要用自定义语言检测，可在此实现 ==========
def detect_language_improved(text):
    """可结合字符统计和 langdetect；也可以直接信任 Whisper 的 result['language']。"""
    chinese_char_count = sum(1 for char in text if '\u4e00' <= char <= '\u9fff')
    english_word_count = len(re.findall(r'\b[a-zA-Z]+\b', text))

    if chinese_char_count > english_word_count + 1:
        return "zh"
    elif english_word_count > chinese_char_count + 1:
        return "en"
    else:
        try:
            lang = detect(text)
            return "zh" if lang.startswith('zh') else "en"
        except:
            return "zh" if chinese_char_count >= english_word_count else "en"


# ========== 第二步：处理单个片段，做“双遍识别”逻辑 ==========
def process_segment(segment_path, start_time, original_audio, quick_model, final_model):
    """
    1) 用 quick_model 做第一遍识别：获取大致语言
    2) 再用 final_model + 确定的 language param 做第二遍精准识别 (word_timestamps=True)
    """
    # --- 第一遍：检测语言（不用词级时间戳，速度更快） ---
    quick_result = quick_model.transcribe(segment_path, language=None, word_timestamps=False)
    # Whisper 会返回 quick_result["language"] 表示它自动检测到的主语言
    # 也可额外对 quick_result["text"] 调用 detect_language_improved(quick_result["text"])

    if "language" in quick_result and quick_result["language"]:
        first_pass_lang = quick_result["language"]  # e.g. 'en' or 'zh'
    else:
        # 如果 Whisper 没有给出，或者你想用自定义的
        first_pass_lang = detect_language_improved(quick_result["text"])

    # 判断大约是中文还是英文
    # 其他情况（如 "ja" 等），你可以留 None 或再次判断
    if first_pass_lang.startswith("en"):
        final_language_param = "en"
    elif first_pass_lang.startswith("zh"):
        final_language_param = "zh"
    else:
        # 如果不是中或英，则可以留 None 让它自由识别
        final_language_param = None

    # --- 第二遍：带 word_timestamps 的精准识别 ---
    whisper_options = {
        "verbose": None,
        "word_timestamps": True,
        "task": "transcribe",
        "suppress_tokens": "",
        "temperature": 0.0,  # 降低随机性
        "condition_on_previous_text": True  # 考虑上下文
    }
    whisper_options["language"] = final_language_param

    final_result = final_model.transcribe(
        segment_path,
        **whisper_options
    )

    # final_result["segments"] 包含真正要输出的分段 + 词级时间戳
    transcription_segments = final_result["segments"]

    processed_segments = []

    for seg in transcription_segments:
        seg_text = seg['text']
        seg_start = seg['start'] + start_time
        seg_end = seg['end'] + start_time

        # 这里再做一遍文本检测或直接沿用 final_language_param
        if final_language_param == "en":
            lang = "en"
        elif final_language_param == "zh":
            lang = "zh"
        else:
            lang = detect_language_improved(seg_text)  # fallback

        words_list = [{
            "word": w["word"],
            "start": w["start"] + start_time,
            "end": w["end"] + start_time,
            "probability": w["probability"]
        } for w in seg.get("words", [])]

        processed_segments.append({
            "Timestamp": f"{seg_start:.2f} - {seg_end:.2f}",
            "Words": words_list,
            "text": seg_text,
            "Language": lang
        })


    return processed_segments


def main():
    # 1) 分段
    segments_info, original_audio = split_audio(AUDIO_FILE, max_size_mb=MAX_SEGMENT_MB)
    print(f"Audio split into {len(segments_info)} segments.")

    # 2) 准备本地 Whisper 模型
    quick_model = whisper.load_model("tiny")
    # --- Final model: 用来做第二遍高精度识别 ---
    final_model = whisper.load_model("large")  # 或 "medium"/"large"

    all_segments = []

    # 3) 逐段处理，每段做两遍识别
    for i, (segment_path, start_sec, end_sec) in enumerate(segments_info, 1):
        print(f"Processing segment {i}/{len(segments_info)}: {segment_path}...")

        processed_seg_list = process_segment(
            segment_path, start_sec, original_audio,
            quick_model,   # 第一遍用
            final_model    # 第二遍用
        )
        all_segments.extend(processed_seg_list)


    # 4) 输出 JSON
    transcription_file = os.path.join(OUTPUT_FOLDER, "Method2_large_WBW.json")
    with open(transcription_file, 'w', encoding='utf-8') as f:
        json.dump(all_segments, f, ensure_ascii=False, indent=2)
    print(f"\nTranscription saved to: {transcription_file}")

    # 5) 清理临时分段文件
    for segment_path, _, _ in segments_info:
        if os.path.exists(segment_path):
            os.remove(segment_path)
    print("Cleanup complete.")


if __name__ == "__main__":
    main()

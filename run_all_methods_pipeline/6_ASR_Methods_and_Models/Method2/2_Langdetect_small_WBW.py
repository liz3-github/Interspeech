import os
import math
import re
from pydub import AudioSegment
from langdetect import detect
import whisper
import json

# Set input audio file and output folder paths
AUDIO_FILE = "{input_file}"
OUTPUT_FOLDER = "{output_folder}"

MAX_SEGMENT_MB = 25  

os.makedirs(OUTPUT_FOLDER, exist_ok=True)

def format_timestamp(seconds: float) -> str:
        minutes = int(seconds // 60)
        seconds = int(seconds % 60)
        return f"{minutes:02d}:{seconds:02d}"

# Step 1: Split audio by file size
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


# Helper: Custom language detection
def detect_language_improved(text):
    """
    Combine character count and langdetect, or simply trust Whisper's result
    """
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


# Step 2: Process a segment with two-pass transcription 
def process_segment(segment_path, start_time, original_audio, quick_model, final_model):
    """
    1) Use quick_model for a fast pass to detect language.
    2) Use final_model with the language parameter for detailed transcription.
    """
    # First pass: quick transcription for language detection
    quick_result = quick_model.transcribe(segment_path, language=None, word_timestamps=False)
    # Whisper returns quick_result["language"] as the detected language.
    # You can also use detect_language_improved(quick_result["text"]).

    if "language" in quick_result and quick_result["language"]:
        first_pass_lang = quick_result["language"]  
    else:
        
        first_pass_lang = detect_language_improved(quick_result["text"])

    # Set language parameter based on the detected language
    if first_pass_lang.startswith("en"):
        final_language_param = "en"
    elif first_pass_lang.startswith("zh"):
        final_language_param = "zh"
    else:
        # If it's not in Chinese or English, equale None to recognize it freely.
        final_language_param = None

    # Second pass: detailed transcription with word timestamps
    whisper_options = {
        "verbose": None,
        "word_timestamps": True,
        "task": "transcribe",
        "suppress_tokens": "",
    }
    whisper_options["language"] = final_language_param

    final_result = final_model.transcribe(
        segment_path,
        **whisper_options
    )

    # final_result["segments"] contains the segments with word timestamps
    transcription_segments = final_result["segments"]

    processed_segments = []

    for seg in transcription_segments:
        seg_text = seg['text']
        seg_start = seg['start'] + start_time
        seg_end = seg['end'] + start_time

        # Use the final language param or fallback to custom detection
        if final_language_param == "en":
            lang = "en"
        elif final_language_param == "zh":
            lang = "zh"
        else:
            lang = detect_language_improved(seg_text) 

        start_str = format_timestamp(seg_start)
        end_str   = format_timestamp(seg_end)

        words_list = []
        for w in seg.get("words", []):
            w_start = w["start"] + start_time
            w_end   = w["end"]   + start_time
            words_list.append({
                "word": w["word"],
                "start": format_timestamp(w_start),
                "end":   format_timestamp(w_end),
                "probability": w["probability"]
            })

        processed_segments.append({
            "Timestamp": f"{start_str} - {end_str}",
            "Words": words_list,
            "text": seg_text,
            "Language": lang
        })
        


    return processed_segments


def main():
    # 1) Split audio into segments
    segments_info, original_audio = split_audio(AUDIO_FILE, max_size_mb=MAX_SEGMENT_MB)
    print(f"Audio split into {len(segments_info)} segments.")

    # 2) Load local Whisper models
    quick_model = whisper.load_model("tiny")
    # Final model: Used for a second high-precision identification
    final_model = whisper.load_model("small")  # small

    all_segments = []

    # 3) Process each segment with two-pass transcription
    for i, (segment_path, start_sec, end_sec) in enumerate(segments_info, 1):
        print(f"Processing segment {i}/{len(segments_info)}: {segment_path}...")

        processed_seg_list = process_segment(
            segment_path, start_sec, original_audio,
            quick_model,   # first pass
            final_model   # second pass
        )
        all_segments.extend(processed_seg_list)


    # 4) Save output JSON
    transcription_file = os.path.join(OUTPUT_FOLDER, "Method2_small_WBW.json")
    with open(transcription_file, 'w', encoding='utf-8') as f:
        json.dump(all_segments, f, ensure_ascii=False, indent=2)
    print(f"\nTranscription saved to: {transcription_file}")

    # 5) Clean up temporary segment files
    for segment_path, _, _ in segments_info:
        if os.path.exists(segment_path):
            os.remove(segment_path)
    print("Cleanup complete.")


if __name__ == "__main__":
    main()

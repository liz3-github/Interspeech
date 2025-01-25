from pydub import AudioSegment
import os
import json
import whisper  

# 设置音频文件路径
AUDIO_FILE = "{input_file}"
OUTPUT_FOLDER = "{output_folder}"

def split_audio(file_path, segment_length=60000, overlap=30000):
    """
    Split audio into overlapping segments
    """
    audio = AudioSegment.from_wav(file_path)
    duration = len(audio)
    segments = []
    for start in range(0, duration, segment_length - overlap):
        end = start + segment_length
        if end > duration:
            end = duration
        segment = audio[start:end]
        segment_path = os.path.join(OUTPUT_FOLDER, f"{os.path.basename(file_path)[:-4]}_part{len(segments)+1}.wav")
        segment.export(segment_path, format="wav")
        segments.append((start, end, segment_path))
    return segments

def transcribe_audio_with_timestamps(model, file_path):
    try:
        whisper_options = {
            "verbose": None,
            "word_timestamps": True,
            "task": "transcribe",
            "suppress_tokens": "",
            "language": None,
            "temperature": 0.0,
            "condition_on_previous_text": True
        }
        result = model.transcribe(file_path, **whisper_options)
        return result
    except Exception as e:
        print(f"An error occurred: {e}")
        return None


def format_timestamp(seconds: float) -> str:
    minutes = int(seconds // 60)
    remaining = int(seconds % 60)
    return f"{minutes:02d}:{remaining:02d}"

def process_transcription(transcription, offset_ms):
    """
    把 Whisper 返回的 segments 加上当前片段的起始偏移量(秒)，
    并格式化成希望的 JSON 结构。
    """
    segs_out = []
    if not transcription or "segments" not in transcription:
        return segs_out

    # offset_ms 是毫秒，所以需要 /1000 才能得到秒
    base_offset_sec = offset_ms / 1000.0

    for seg in transcription["segments"]:
        local_start = seg["start"]  # Whisper给出的该段在文件中(秒)
        local_end   = seg["end"]

        # 计算在原始大音频里的绝对时间(秒)
        abs_start = base_offset_sec + local_start
        abs_end   = base_offset_sec + local_end

        # 组装一个区间字符串，比如 "12.34 - 18.90"
        timestamp_str = f"{abs_start:.2f} - {abs_end:.2f}"

        # 再做一个 mm:ss 的字符串（可选）
        start_str = format_timestamp(abs_start)
        end_str   = format_timestamp(abs_end)

        # 处理词级别时间戳
        words_list = []
        if "words" in seg:  # 当 word_timestamps=True 时才会有 "words"
            for w in seg["words"]:
                w_abs_start = base_offset_sec + w["start"]
                w_abs_end   = base_offset_sec + w["end"]
                words_list.append({
                    "word": w["word"],
                    "start": round(w_abs_start, 2),
                    "end":   round(w_abs_end,   2)
                })

        # 把该段信息添加到输出列表
        segs_out.append({
            "Timestamp": timestamp_str,      # 浮点秒数区间
            "start": start_str,              # mm:ss
            "end": end_str,                  # mm:ss
            "Text": seg["text"].strip(),     # 段落文本
            "Speaker": "",
            "Words": words_list,
            "Language": transcription.get("language", "")
        })

    return segs_out

def transcribe_audio_segments(model, segments):
    all_transcriptions = []
    for i, (start_ms, end_ms, segment_path) in enumerate(segments, 1):
        print(f"Transcribing segment {i}/{len(segments)}: {segment_path}")
        # 调用时，传入model和segment_path两个参数
        transcription = transcribe_audio_with_timestamps(model, segment_path)
        if transcription:
            processed_segs = process_transcription(transcription, start_ms)
            all_transcriptions.extend(processed_segs)
        else:
            print(f"Transcription failed for segment {i}")
    return all_transcriptions



def merge_segments(segments):
    """
    Merge adjacent segments of the same language
    and also merge their word lists.
    假设 segments 里字段是:
    {
      "Timestamp": "12.00 - 18.90",
      "start": "00:12",
      "end": "00:18",
      "Text": "...",
      "Words": [...],
      "Language": "en",
      ...
    }
    """
    def timestamp_to_seconds(ts: str) -> float:
        # 这里 ts 类似 "00:12" => 0分12秒
        mm, ss = ts.split(':')
        mm = float(mm)
        ss = float(ss)
        return mm*60 + ss

    merged = []
    for seg in segments:
        if not merged:
            merged.append(seg)
        else:
            last_seg = merged[-1]

            # 计算两段之间的时间差(单位秒)
            gap = timestamp_to_seconds(seg['start']) - timestamp_to_seconds(last_seg['end'])
            same_lang = (last_seg['Language'] == seg['Language'])

            # 如果语言相同且它们的时间戳紧邻(例如间隔<0.1s)，我们把它合并成一个段
            if same_lang and gap < 0.1:
                # 1) 合并文本
                last_seg['end'] = seg['end']
                # 更新 Timestamp: 把最左的 start 不变, end 改成 seg['Timestamp'] 的右值
                old_ts_left = last_seg['Timestamp'].split(' - ')[0]
                new_ts_right = seg['Timestamp'].split(' - ')[1]
                last_seg['Timestamp'] = f"{old_ts_left} - {new_ts_right}"

                last_seg['Text'] += " " + seg['Text']

                # 2) 合并单词列表
                if 'Words' in last_seg and 'Words' in seg:
                    last_seg['Words'].extend(seg['Words'])
            else:
                merged.append(seg)
    return merged


def main():
    # 1) 拆分音频
    print("Splitting audio into overlapping segments...")
    audio_segments = split_audio(AUDIO_FILE, segment_length=60000, overlap=30000)
    print(f"Audio split into {len(audio_segments)} segments.")

    # 2) 加载一次 Whisper 模型
    print("Loading whisper 'small' model...")
    model = whisper.load_model("small")

    # 3) 对各分段做转录并汇总
    all_segments = transcribe_audio_segments(model, audio_segments)

    # 4) 合并相邻同语言的段
    merged_segments = merge_segments(all_segments)

    # 5) 输出到 JSON
    if merged_segments:
        transcription_file = os.path.join(OUTPUT_FOLDER, "Method3_Model_small_WBW.json")
        with open(transcription_file, 'w', encoding='utf-8') as f:
            json.dump(merged_segments, f, ensure_ascii=False, indent=2)
        print(f"Transcription saved to {transcription_file}")
    else:
        print("No segments to save. Transcription might have failed.")

    # 6) 清理临时文件
    print("Cleaning up temporary files...")
    for _, _, segment_path in audio_segments:
        if os.path.exists(segment_path):
            os.remove(segment_path)
    print("Cleanup complete.")

if __name__ == "__main__":
    main()
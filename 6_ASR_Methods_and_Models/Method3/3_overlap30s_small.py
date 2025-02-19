from pydub import AudioSegment
import os
import json
import whisper  

# Set input audio file and output folder paths
AUDIO_FILE = "{input_file}"
OUTPUT_FOLDER = "{output_folder}"

def format_timestamp(seconds: float) -> str:
        minutes = int(seconds // 60)
        seconds = int(seconds % 60)
        return f"{minutes:02d}:{seconds:02d}"

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
    Add the segment's starting offset (in ms) to Whisper's segments,
    and format the result into our JSON structure.
    """
    segs_out = []
    if not transcription or "segments" not in transcription:
        return segs_out

    # Convert offset from ms to seconds
    base_offset_sec = offset_ms / 1000.0

    for seg in transcription["segments"]:
        local_start = seg["start"]  
        local_end   = seg["end"]

        # Calculate absolute times based on the original audio
        abs_start = base_offset_sec + local_start
        abs_end   = base_offset_sec + local_end

        # Create a timestamp range string ("12.34 - 18.90")
        timestamp_str = f"{abs_start:.2f} - {abs_end:.2f}"

        
        start_str = format_timestamp(abs_start)
        end_str   = format_timestamp(abs_end)

        # Process word-level timestamps
        words_list = []
        if "words" in seg:  #when word_timestamps=True -> "words"
            for w in seg["words"]:
                w_abs_start = base_offset_sec + w["start"]
                w_abs_end   = base_offset_sec + w["end"]
                words_list.append({
                    "word": w["word"],
                    "start": format_timestamp(w_abs_start),
                    "end":   format_timestamp(w_abs_end)
                })

        
        segs_out.append({
            "Timestamp": timestamp_str,      # Absolute time range in seconds
            "start": start_str,              # mm:ss
            "end": end_str,                  # mm:ss
            "Text": seg["text"].strip(),     # Transcribed text
            "Speaker": "",
            "Words": words_list,
            "Language": transcription.get("language", "")
        })

    return segs_out

def transcribe_audio_segments(model, segments):
    """
    Transcribe each audio segment, adjust local timestamps to global time,
    and combine all results.
    """
    all_transcriptions = []
    for i, (start_ms, end_ms, segment_path) in enumerate(segments, 1):
        print(f"Transcribing segment {i}/{len(segments)}: {segment_path}")
        
        transcription = transcribe_audio_with_timestamps(model, segment_path)
        if transcription:
            processed_segs = process_transcription(transcription, start_ms)
            all_transcriptions.extend(processed_segs)
        else:
            print(f"Transcription failed for segment {i}")
    return all_transcriptions



def merge_segments(segments):
    """
    Merge adjacent segments with the same language and combine their word lists.
    Assumes segments have the following fields:
      {
        "Timestamp": "12.00 - 18.90",
        "start": "00:12",
        "end": "00:18",
        "Text": "...",
        "Words": [...],
        "Language": "en",
      }
    """
    def timestamp_to_seconds(ts: str) -> float:
        # Convert a timestamp "mm:ss" to total seconds
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

            # Calculate the time difference between two segments (in seconds)
            gap = timestamp_to_seconds(seg['start']) - timestamp_to_seconds(last_seg['end'])
            same_lang = (last_seg['Language'] == seg['Language'])

            #  Merge if languages are the same and segments are nearly contiguous
            if same_lang and gap < 0.1:
                # 1) 合并文本
                last_seg['end'] = seg['end']
                # Update Timestamp: leave the leftmost start unchanged and change the end to the right value of seg['Timestamp'].
                old_ts_left = last_seg['Timestamp'].split(' - ')[0]
                new_ts_right = seg['Timestamp'].split(' - ')[1]
                last_seg['Timestamp'] = f"{old_ts_left} - {new_ts_right}"

                last_seg['Text'] += " " + seg['Text']

                # Merge word lists
                if 'Words' in last_seg and 'Words' in seg:
                    last_seg['Words'].extend(seg['Words'])
            else:
                merged.append(seg)
    return merged


def main():
    # 1) Split audio into overlapping segments
    print("Splitting audio into overlapping segments...")
    audio_segments = split_audio(AUDIO_FILE, segment_length=60000, overlap=30000)
    print(f"Audio split into {len(audio_segments)} segments.")

    # 2) Load the Whisper 'small' model
    print("Loading whisper 'small' model...")
    model = whisper.load_model("small")

    # 3) Transcribe each segment and combine results
    all_segments = transcribe_audio_segments(model, audio_segments)

    # 4) Merge adjacent segments with the same language
    merged_segments = merge_segments(all_segments)

    if merged_segments:
        transcription_file = os.path.join(OUTPUT_FOLDER, "Method3_small_WBW.json")
        with open(transcription_file, 'w', encoding='utf-8') as f:
            json.dump(merged_segments, f, ensure_ascii=False, indent=2)
        print(f"Transcription saved to {transcription_file}")
    else:
        print("No segments to save. Transcription might have failed.")

    # 6) Clean up temporary segment files
    print("Cleaning up temporary files...")
    for _, _, segment_path in audio_segments:
        if os.path.exists(segment_path):
            os.remove(segment_path)
    print("Cleanup complete.")

if __name__ == "__main__":
    main()
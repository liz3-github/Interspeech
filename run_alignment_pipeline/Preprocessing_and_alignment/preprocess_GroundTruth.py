import os
import glob
from typing import List

# Convert "mm:ss" string to seconds.
def parse_time_to_seconds(t: str) -> int:
    try:
        mm, ss = t.strip().split(':')
        return int(mm) * 60 + int(ss)
    except:
        return -1

# Split a timestamp range (e.g., "mm:ss - mm:ss") into start and end seconds.
def parse_timestamp(ts_str: str):
    parts = ts_str.split('-')
    if len(parts) != 2:
        return None, None
    
    start_str = parts[0].strip()
    end_str   = parts[1].strip()
    start_sec = parse_time_to_seconds(start_str)
    end_sec   = parse_time_to_seconds(end_str)
    if start_sec == -1 or end_sec == -1:
        return None, None
    
    return start_sec, end_sec

# Create a timestamp string from start and end seconds.
def build_timestamp(start_sec: int, end_sec: int) -> str:
    start_str = f"{start_sec // 60:02d}:{start_sec % 60:02d}"
    end_str   = f"{end_sec // 60:02d}:{end_sec % 60:02d}"
    return f"{start_str} - {end_str}"

# Merge segments if the end of one equals the start of the next.
def merge_adjacent_segments(segments: List[dict]) -> List[dict]:
    if not segments:
        return []
    
    merged = [segments[0]]
    for i in range(1, len(segments)):
        current = segments[i]
        last    = merged[-1]

        last_start_sec, last_end_sec = parse_timestamp(last["Timestamp"])
        curr_start_sec, curr_end_sec = parse_timestamp(current["Timestamp"])

        if (
            last_start_sec is not None and last_end_sec is not None and
            curr_start_sec is not None and curr_end_sec is not None and
            last_end_sec == curr_start_sec
        ):
            # Merge timestamp.
            new_ts = build_timestamp(last_start_sec, curr_end_sec)
            last["Timestamp"] = new_ts

            # Merge text.
            last_text = last.get("Text", "")
            curr_text = current.get("Text", "")
            if curr_text:
                if last_text:
                    last["Text"] = last_text + "\n" + curr_text
                else:
                    last["Text"] = curr_text

            # Merge Speaker.
            if "Speaker" in current:
                if "Speaker" in last:
                    if current["Speaker"] != last["Speaker"]:
                        last["Speaker"] = f"{last['Speaker']}, {current['Speaker']}"
                else:
                    last["Speaker"] = current["Speaker"]

            # Merge Language
            if "Language" in current:
                if "Language" in last:
                    if current["Language"] != last["Language"]:
                        last["Language"] = f"{last['Language']}, {current['Language']}"
                else:
                    last["Language"] = current["Language"]

        else:
            merged.append(current)

    return merged

# Read, filter, and merge transcript entries from a file.
def filter_transcript(input_file: str, output_file: str) -> None:
    segments = []
    current_entry = {}
    text_buffer   = []

    with open(input_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.rstrip('\n')
            if line.startswith("Timestamp:"):
                if current_entry:
                    if text_buffer:
                        current_entry['Text'] = '\n'.join(text_buffer)
                    segments.append(current_entry)
                ts_val = line.split("Timestamp:")[1].strip()
                current_entry = {"Timestamp": ts_val}
                text_buffer   = []
            elif line.startswith("OnsetTime:") and current_entry:
                val = line.split("OnsetTime:")[1].strip()
                current_entry['OnsetTime'] = val
            elif line.startswith("Text:") and current_entry:
                txt = line[5:].strip()
                text_buffer = [txt]
            elif line.startswith("Speaker:") and current_entry:
                spk = line.split("Speaker:")[1].strip()
                current_entry['Speaker'] = spk
            elif line.startswith("Lang:") and current_entry:
                lang = line.split("Lang:")[1].strip()
                current_entry['Language'] = lang
            else:
                if current_entry is not None and text_buffer is not None and line.strip():
                    text_buffer.append(line.strip())

    if current_entry:
        if text_buffer:
            current_entry['Text'] = '\n'.join(text_buffer)
        segments.append(current_entry)

    # Remove segments with empty text.
    filtered = []
    for seg in segments:
        text_val = seg.get('Text','').strip()
        if text_val:
            filtered.append(seg)

    # Merge adjacent paragraphs
    merged_segments = merge_adjacent_segments(filtered)

    with open(output_file, 'w', encoding='utf-8') as fout:
        for seg in merged_segments:
            fout.write(f"Timestamp: {seg.get('Timestamp','')}\n")
            if 'OnsetTime' in seg:
                fout.write(f"OnsetTime: {seg.get('OnsetTime','')}\n")
            if 'Text' in seg:
                text_block = seg['Text']
                fout.write(f"Text: {text_block}\n")
            if 'Speaker' in seg:
                fout.write(f"Speaker: {seg.get('Speaker','')}\n")
            if 'Language' in seg:
                fout.write(f"Lang: {seg.get('Language','')}\n")
            fout.write("\n")

    print(f"[OK] {input_file} => {output_file}")


# Batch process all .txt files in the "annotated" folder.
if __name__ == '__main__':
    annotated_dir = os.path.join(os.path.dirname(__file__), "annotated")
    output_dir    = os.path.join(os.path.dirname(__file__), "annotated_merged")
    os.makedirs(output_dir, exist_ok=True)

    txt_files = glob.glob(os.path.join(annotated_dir, "*.txt"))

    for input_path in txt_files:
        filename = os.path.basename(input_path)             
        base, ext = os.path.splitext(filename)              
        output_filename = base + ".txt"    
        output_path = os.path.join(output_dir, output_filename)

        filter_transcript(input_path, output_path)

    print("\nAll transcripts processed!")

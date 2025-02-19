import json
import csv
import logging
from typing import List, Dict, Tuple

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

class TimeSegment:
    """Time segment (start, end)."""
    def __init__(self, start: float, end: float):
        self.start = start
        self.end   = end

    @staticmethod
    def format_time(seconds: float) -> str:
        """Convert seconds to MM:SS.SS format."""
        if seconds == '':
            return ''
        m = int(seconds // 60)
        s = seconds % 60
        return f"{m:02d}:{s:05.2f}"


class Word:
    """Word-level transcript."""
    def __init__(self, text: str, start: float, end: float,
                 speaker: str = "", language: str = ""):
        self.text     = text
        self.start    = start
        self.end      = end
        self.speaker  = speaker
        self.language = language

    @property
    def midpoint(self) -> float:
        return (self.start + self.end) / 2


class TranscriptSegment:
    """Manual transcript segment."""
    def __init__(self, start: float, end: float, text: str,
                 speaker: str = "", language: str = "", onset: str = ""):
        self.start    = start
        self.end      = end
        self.text     = text
        self.speaker  = speaker
        self.language = language
        self.onset    = onset

        # Machine words finally matched for this paragraph
        self.words: List[Word] = []

    def get_original_timestamp(self) -> str:
        start_str = TimeSegment.format_time(self.start)
        end_str   = TimeSegment.format_time(self.end)
        return f"{start_str} - {end_str}"


# Utility Function
def convert_to_seconds(timestr: str) -> float:
    """Convert time string (e.g., '00:03', '1.23') to seconds."""
    timestr = timestr.strip()
    if not timestr:
        return 0.0
    try:
        if ':' in timestr:
            parts = timestr.split(':')
            parts = [float(p) for p in parts]
            if len(parts) == 2:
                # MM:SS
                mm, ss = parts
                return mm * 60 + ss
            elif len(parts) == 3:
                hh, mm, ss = parts
                return hh*3600 + mm*60 + ss
        return float(timestr)
    except:
        return 0.0


def load_machine_transcript(json_file: str) -> List[Dict]:
    """Load machine transcript JSON."""
    with open(json_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data

def load_manual_transcript(txt_file: str) -> List[Dict]:
    """
    Load manual transcript file.
    Returns list of dicts with keys: Timestamp, OnsetTime, Text, Speaker, Language.
    """
    segments = []
    current_entry = None
    text_buffer   = []

    with open(txt_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.rstrip('\n')
            if line.startswith("Timestamp:"):
                # If there is already a current_entry, write the buffer first.
                if current_entry:
                    if text_buffer:
                        current_entry['Text'] = '\n'.join(text_buffer)
                    segments.append(current_entry)
                
                ts_val = line.split("Timestamp:")[1].strip()
                current_entry = {"Timestamp": ts_val}
                text_buffer = []

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
                
                if current_entry and text_buffer is not None and line.strip():
                    text_buffer.append(line.strip())


    if current_entry:
        if text_buffer:
            current_entry['Text'] = '\n'.join(text_buffer)
        segments.append(current_entry)

    return segments

# save results
def write_csv_output(results: List[Dict], output_path: str):
    """将对齐结果写入CSV"""
    fieldnames = [
        'machine_onset',
        'machine_end',
        'machine_timestamp',
        'machine_text',
        'machine_speaker',
        'machine_language',
        'human_onset',
        'human_timestamp',
        'human_text',
        'speaker',
        'language',
        'word_count',
        'overlaps',
        'human_start_seconds'
    ]

    with open(output_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

# Core Alignment Class 
class TranscriptAligner:
    def __init__(self):
        pass

    def prepare_machine_words(self, machine_data: List[Dict]) -> List[Word]:
        """Extract and sort words from machine transcript."""
        words = []
        for seg_obj in machine_data:
            seg_speaker  = seg_obj.get("Speaker","")
            seg_language = seg_obj.get("Language","")

            for wobj in seg_obj.get("Words", []):
                wstart_sec = convert_to_seconds(wobj.get("start","0"))
                wend_sec   = convert_to_seconds(wobj.get("end","0"))
                text       = wobj.get("word","").strip()

                w = Word(
                    text    = text,
                    start   = wstart_sec,
                    end     = wend_sec,
                    speaker = seg_speaker,
                    language= seg_language
                )
                words.append(w)

        # Sorted by start time
        words.sort(key=lambda w: w.start)
        return words

    def prepare_manual_segments(self, manual_data: List[Dict]) -> List[TranscriptSegment]:
        """Convert manual transcript dicts to TranscriptSegment objects."""
        segments = []
        for entry in manual_data:
            ts = entry.get("Timestamp","")
            start, end = 0.0, 0.0
            try:
                start_str, end_str = ts.split(' - ')
                start = convert_to_seconds(start_str)
                end   = convert_to_seconds(end_str)
            except:
                pass

            seg = TranscriptSegment(
                start    = start,
                end      = end,
                text     = entry.get("Text","").strip(),
                speaker  = entry.get("Speaker","").strip(),
                language = entry.get("Language","").strip(),
                onset    = entry.get("OnsetTime","").strip()
            )
            segments.append(seg)

        segments.sort(key=lambda s: s.start)
        return segments

    def is_sentence_end(self, text: str) -> bool:
        end_punctuations = {'.', '。', '!', '！', '?', '？', '...', '…'}
        return any(p in text for p in end_punctuations)

    # (1) First alignment: return (segments, unmatched words)
    def strict_align(self, manual_data: List[Dict], machine_data: List[Dict]) -> Tuple[List[TranscriptSegment], List[Word]]:
        segs = self.prepare_manual_segments(manual_data)
        words_all = self.prepare_machine_words(machine_data)
        
        i = 0  # manual segments index
        j = 0  # machine words index
        
        unmatched_words = []
        
        while i < len(segs) and j < len(words_all):
            seg = segs[i]
            word = words_all[j]
            
            # When word at end of paragraph == start of word (or very small gap)
            if abs(word.start - seg.end) < 1e-6:
                # Get the previous word
                prev_word = words_all[j-1] if j > 0 else None
                if prev_word and self.is_sentence_end(prev_word.text):
                    # Previous word with ending punctuation => end of sentence => next paragraph
                    i += 1
                    continue
                else:
                    seg.words.append(word)
                    j += 1

            elif word.end < seg.start:
                # machine words before paragraphs => mismatch
                unmatched_words.append(word)
                j += 1

            elif word.start > seg.end:
                # Machine words after the paragraph => end of this paragraph
                i += 1

            else:
                # with actual overlap => match to this paragraph
                seg.words.append(word)
                j += 1
        
        # If there are words left, they're unmatched
        while j < len(words_all):
            unmatched_words.append(words_all[j])
            j += 1

        # Print first alignment
        print("====== Unmatched Words (first align) ======")
        for uw in unmatched_words:
            print(
                f"Text: {uw.text}, "
                f"Start: {TimeSegment.format_time(uw.start)}, "
                f"End: {TimeSegment.format_time(uw.end)}, "
                f"Speaker: {uw.speaker}, "
                f"Lang: {uw.language}"
            )
        
        return segs, unmatched_words

    # (2) Second alignment on unmatched words
    def second_align(self,
                     segments: List[TranscriptSegment],
                     unmatched_words: List[Word]) -> List[Word]:
        """
        Double pointer tries to align again: word.end < seg.start => word remains unmatched.
          - If word.end < seg.start, word is earlier than current paragraph => word is still unmatched, j++
          - if word.start > seg.end, the paragraph is too early, i++
          - otherwise there is an overlap => merge into the paragraph, j++
        Returns: still can't match any word in any paragraph
        """
        i = 0
        j = 0

        while i < len(segments) and j < len(unmatched_words):
            seg = segments[i]
            w = unmatched_words[j]

            if w.end < seg.start:
                # words before the start of the current paragraph => considered still unmatched
                j += 1

            elif w.start > seg.end:
                # Words later than current paragraph => paragraph too early, i++
                i += 1

            else:
                # with time overlap => merge into current paragraph
                seg.words.append(w)
                # Optional: if you want the paragraph to cover the word time
                seg.start = min(seg.start, w.start)
                seg.end   = max(seg.end,   w.end)
                j += 1

        # unmatched_words[j:]  unmatched
        still_unmatched = unmatched_words[j:]
        return still_unmatched

    # Convert segments to CSV rows
    def convert_segments_to_results(self, segments: List[TranscriptSegment]) -> List[Dict]:
        results = []
        for seg in segments:
            row = {
                'human_onset': seg.onset,
                'human_timestamp': seg.get_original_timestamp(),
                'human_text': seg.text,
                'speaker': seg.speaker,
                'language': seg.language,
                'human_start_seconds': seg.start
            }

            if seg.words:
                w_start = min(w.start for w in seg.words)
                w_end   = max(w.end for w in seg.words)
                machine_text    = ' '.join(w.text for w in seg.words)
                #machine_speaker = seg.words[0].speaker
                #machine_lang    = seg.words[0].language
                speaker_set  = set(w.speaker for w in seg.words if w.speaker)
                language_set = set(w.language for w in seg.words if w.language)
                machine_speaker = ", ".join(sorted(speaker_set))
                machine_lang    = ", ".join(sorted(language_set))
                
                row.update({
                    'machine_onset': TimeSegment.format_time(w_start),
                    'machine_end':   TimeSegment.format_time(w_end),
                    'machine_timestamp': f"{TimeSegment.format_time(w_start)} - {TimeSegment.format_time(w_end)}",
                    'machine_text': machine_text,
                    'machine_speaker': machine_speaker,
                    'machine_language': machine_lang,
                    'word_count': len(seg.words),
                    'overlaps': any((w.start < seg.start or w.end > seg.end) for w in seg.words)
                })
            else:
                row.update({
                    'machine_onset': '',
                    'machine_end': '',
                    'machine_timestamp': '',
                    'machine_text': '',
                    'machine_speaker': '',
                    'machine_language': '',
                    'word_count': 0,
                    'overlaps': False
                })

            results.append(row)

        # Sorted by manual paragraphstart
        results.sort(key=lambda r: r['human_start_seconds'])
        return results


def main():
    import sys
    import os
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--manual_file", required=True, help="Path to manual txt")
    parser.add_argument("--machine_file", required=True, help="Path to machine json")
    parser.add_argument("--output_csv", required=True, help="Output CSV path")
    args = parser.parse_args()

    manual_file = args.manual_file
    machine_file= args.machine_file
    output_file = args.output_csv

    if not os.path.exists(manual_file):
        print("[ERROR] manual_file not found:", manual_file)
        sys.exit(1)
    if not os.path.exists(machine_file):
        print("[ERROR] machine_file not found:", machine_file)
        sys.exit(1)

    try:
        # Load transcripts
        manual_data  = load_manual_transcript(manual_file)
        machine_data = load_machine_transcript(machine_file)

        aligner = TranscriptAligner()

        # First alignment round
        segs, unmatched_words = aligner.strict_align(manual_data, machine_data)
        print(f"[DEBUG] {len(unmatched_words)} words unmatched after first align.")

        # Second alignment round
        still_unmatched = aligner.second_align(segs, unmatched_words)
        print(f"[DEBUG] {len(still_unmatched)} words still unmatched after second align.")

        # Convert segments to CSV rows and write output
        results = aligner.convert_segments_to_results(segs)
        write_csv_output(results, output_file)

        print(f"[INFO] Done! {len(results)} segments written to {output_file}.")

    except Exception as e:
        logger.error(f"Error: {e}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    main()
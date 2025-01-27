import csv
import argparse
import json
from datetime import datetime
from typing import List, Dict, Any, Tuple, Optional
import logging

logging.basicConfig(
    filename='alignment_debug.log',
    level=logging.DEBUG,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


class TimeSegment:
    """表示一个时间段"""
    def __init__(self, start: float, end: float):
        self.start = start
        self.end = end

    def get_overlap_duration(self, other: 'TimeSegment') -> float:
        """计算与另一个时间段的重叠时长"""
        if self.end < other.start or self.start > other.end:
            return 0
        return min(self.end, other.end) - max(self.start, other.start)

    @staticmethod
    def format_time(seconds: float) -> str:
        """将秒数转换为MM:SS.SS格式"""
        if seconds == '':
            return ''
        minutes = int(float(seconds) // 60)
        remaining_seconds = float(seconds) % 60
        return f"{minutes:02d}:{remaining_seconds:05.2f}"

class Word:
    """表示一个单词"""
    def __init__(self, text: str, start: float, end: float,
                 speaker: str = "", language: str = ""):
        self.text = text
        self.segment = TimeSegment(start, end)
        self.duration = end - start
        self.speaker = speaker
        self.language = language

        
    @property
    def midpoint(self) -> float:
        """获取单词的时间中点"""
        return (self.segment.start + self.segment.end) / 2

class TranscriptSegment:
    """表示转录文本的一个片段"""
    def __init__(self, start: float, end: float, text: str, 
                 speaker: str = "", language: str = "", onset: str = ""):
        self.time_segment = TimeSegment(start, end)
        self.text = text
        self.speaker = speaker
        self.language = language
        self.onset = onset
        self.words: List[Word] = []
        self.locked = False  # 表示此段是否已锁定(整句匹配完成)

    def get_original_timestamp(self) -> str:
        """获取原始时间戳格式"""
        start_time = TimeSegment.format_time(self.time_segment.start)
        end_time = TimeSegment.format_time(self.time_segment.end)
        return f"{start_time} - {end_time}"

def convert_to_seconds(time_str: str) -> float:
    """将时间字符串转换为秒数"""

    time_str = time_str.replace('：', ':')
    
    try:
        if isinstance(time_str, (int, float)):
            return float(time_str)
        if ':' in time_str:
            minutes, seconds = time_str.split(':')
            return int(minutes) * 60 + float(seconds)
        return float(time_str)
    except:
        return 0.0

def load_manual_transcript(file_path: str) -> List[Dict]:
    """加载人工转录数据，保持Text字段的原始格式"""
    manual_data = []
    current_entry = None
    text_lines = []  # 用于收集Text的多行内容
    
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            line = line.strip()
            if line.startswith('Timestamp:'):
                if current_entry:
                    if text_lines:  # 将收集的text_lines合并到当前entry
                        current_entry['Text'] = '\n'.join(text_lines)
                    manual_data.append(current_entry)
                current_entry = {'Timestamp': line.split('Timestamp:')[1].strip()}
                
                
                text_lines = []
            elif line.startswith('OnsetTime:') and current_entry:
                if text_lines:
                    current_entry['Text'] = '\n'.join(text_lines)
                    text_lines = []
                current_entry['OnsetTime'] = line.split('OnsetTime:')[1].strip()
            elif line.startswith('Text:') and current_entry:
                text_lines = [line[5:].strip()] 
            elif line.startswith('Speaker:') and current_entry:
                if text_lines:
                    current_entry['Text'] = '\n'.join(text_lines)
                    text_lines = []
                current_entry['Speaker'] = line.split('Speaker:')[1].strip()
            elif line.startswith('Lang:') and current_entry:
                if text_lines:
                    current_entry['Text'] = '\n'.join(text_lines)
                    text_lines = []
                current_entry['Language'] = line.split('Lang:')[1].strip()
            elif line and current_entry and text_lines:
                text_lines.append(line)
    
    if current_entry:
        if text_lines:
            current_entry['Text'] = '\n'.join(text_lines)
        manual_data.append(current_entry)
    
    return manual_data

def load_machine_transcript(file_path: str) -> List[Dict]:
    """加载机器转录数据"""
    with open(file_path, 'r', encoding='utf-8') as file:
        return json.load(file)


class TranscriptAligner:
    """转录文本对齐类"""
    def __init__(self):
        self._setup_rules()
        self.assigned_words = set()  # 跟踪已分配词

    def _setup_rules(self):
        self.rules = [
            self._basic_rule,     
            self._boundary_rule,  
            self._midpoint_rule   
        ]

    def _is_part_of_sentence(self, word: Word, segment: TranscriptSegment) -> bool:
        if not segment.words:
            logger.debug(f"_is_part_of_sentence: No words in segment for word: {word.text}")
            return False
                
        prev_text = ' '.join(w.text for w in segment.words)
        sentence_end_marks = {'.', '?', '!'}
        has_end_mark = any(prev_text.strip().endswith(mark) for mark in sentence_end_marks)
        logger.debug(f"_is_part_of_sentence: Checking {word.text}, has_end_mark={has_end_mark}")
        return not has_end_mark

    def _basic_rule(self, word: Word, segments: List[TranscriptSegment]) -> Optional[TranscriptSegment]:
        boundary_tolerance = 0.5
        logger.debug(f"[BASIC_RULE] Checking word '{word.text}' time {word.segment.start}-{word.segment.end}")
        for i, segment in enumerate(segments):
            logger.debug(f"  Segment {i}: {segment.time_segment.start}-{segment.time_segment.end}")
            if i < len(segments) - 1:
                next_segment = segments[i + 1]
                current_boundary = segment.time_segment.end
                if (abs(word.segment.start - current_boundary) <= boundary_tolerance or
                    abs(word.segment.end - current_boundary) <= boundary_tolerance or
                    word.segment.start < current_boundary < word.segment.end or
                    word.segment.start == current_boundary or
                    word.segment.start == next_segment.time_segment.start):
                    logger.debug(f"[BASIC_RULE] Boundary word detected for '{word.text}', returning None")
                    return None

            if (segment.time_segment.start < word.segment.start < 
                segment.time_segment.end - boundary_tolerance):
                logger.debug(f"[BASIC_RULE] Word '{word.text}' fits inside {segment.time_segment.start}-{segment.time_segment.end}")
                return segment
                    
        logger.debug(f"[BASIC_RULE] No suitable segment found for '{word.text}', returning None.")
        return None

    def _boundary_rule(self, word: Word, segments: List[TranscriptSegment]) -> Optional[TranscriptSegment]:
        logger.debug(f"[BOUNDARY_RULE] Checking word '{word.text}' time {word.segment.start}-{word.segment.end}")
        boundary_tolerance = 0.5
        for i in range(len(segments)-1):
            current_segment = segments[i]
            next_segment = segments[i+1]
            boundary = current_segment.time_segment.end
            logger.debug(f"  Checking boundary between segment {i} & {i+1}: boundary={boundary}")
            if (abs(word.segment.start - boundary) <= boundary_tolerance or
                abs(word.segment.end - boundary) <= boundary_tolerance or
                (word.segment.start < boundary < word.segment.end)):

                logger.debug("  Word is near a boundary, checking part_of_sentence...")
                if self._is_part_of_sentence(word, current_segment):
                    logger.debug(f"[BOUNDARY_RULE] Word '{word.text}' is part of current segment sentence.")
                    return current_segment
                    
                dist_to_current = abs(boundary - word.segment.start)
                dist_to_next = abs(word.segment.end - boundary)
                logger.debug(f"  dist_to_current={dist_to_current}, dist_to_next={dist_to_next}, midpoint={word.midpoint}")

                if abs(dist_to_current - dist_to_next) < 0.1:
                    chosen = current_segment if word.midpoint <= boundary else next_segment
                    logger.debug(f"[BOUNDARY_RULE] distances nearly equal, chosen segment: {chosen.time_segment.start}-{chosen.time_segment.end}")
                    return chosen
                    
                chosen = current_segment if dist_to_current < dist_to_next else next_segment
                logger.debug(f"[BOUNDARY_RULE] Chosen segment by distance: {chosen.time_segment.start}-{chosen.time_segment.end}")
                return chosen

        logger.debug(f"[BOUNDARY_RULE] No boundary condition met for '{word.text}', returning None.")
        return None

    def _midpoint_rule(self, word: Word, segments: List[TranscriptSegment]) -> Optional[TranscriptSegment]:
        logger.debug(f"[MIDPOINT_RULE] Checking word '{word.text}' midpoint {word.midpoint}")
        for segment in segments:
            logger.debug(f"  Segment: {segment.time_segment.start}-{segment.time_segment.end}")
            if (segment.time_segment.start <= word.midpoint <= segment.time_segment.end):
                logger.debug(f"[MIDPOINT_RULE] Word '{word.text}' fits by midpoint in segment {segment.time_segment.start}-{segment.time_segment.end}")
                return segment
        logger.debug(f"[MIDPOINT_RULE] No segment fits '{word.text}' by midpoint, returning None.")
        return None

    def assign_word(self, word: Word, segments: List[TranscriptSegment]) -> Optional[TranscriptSegment]:
        logger.debug(f"[ASSIGN_WORD] Attempting to assign word '{word.text}' ({word.segment.start}-{word.segment.end}) to segments...")
        for r, rule in enumerate(self.rules):
            logger.debug(f"  Trying rule {r}: {rule.__name__}")
            result = rule(word, segments)
            if result is not None:
                logger.debug(f"[ASSIGN_WORD] Word '{word.text}' assigned to segment {result.time_segment.start}-{result.time_segment.end} by {rule.__name__}")
                return result
        logger.debug(f"[ASSIGN_WORD] No rule could assign word '{word.text}', returning None.")
        return None
    
    

    def align(self, manual_data: List[Dict], machine_data: List[Dict]) -> List[Dict]:
        results = []
        all_words = self._prepare_words(machine_data)
        assigned_words = set()
        
        # 第一轮对齐结果和未匹配数据的临时存储结构
        first_round_results = []

        
        # ========== 第一轮对齐（整句级别） ==========
        for entry in manual_data:
            if 'Timestamp' not in entry:
                start, end = self._parse_timestamp(entry['Timestamp'])
                logger.debug(f"Manual entry time range: {start}-{end}")
                continue

            start, end = self._parse_timestamp(entry['Timestamp'])
            current_segment = TranscriptSegment(
                start=start,
                end=end,
                text=entry.get('Text', '').strip(),
                speaker=entry.get('Speaker', '').strip(),
                language=entry.get('Language', '').strip(),
                onset=entry.get('OnsetTime', '').strip()
            )

            first_pass_words = []
            unmatched_words = []
            sentence_started = False

            # 第一轮逻辑：只尝试整句匹配
            for word in all_words:
                word_key = f"{word.text}_{word.segment.start:.2f}_{word.segment.end:.2f}"
                if word_key in assigned_words:
                    continue
                
                # 聚焦整句对齐
                if not sentence_started:
                    # 尝试开始新的句子
                    if (abs(float(word.segment.start) - start) <= 1) or (word.segment.start > start and word.segment.start < end):
                        sentence_started = True
                        first_pass_words.append(Word(
                            text=word.text.strip(),
                            start=float(word.segment.start),
                            end=float(word.segment.end),
                            speaker=word.speaker,
                            language=word.language
                        ))
                    else:
                        # 暂时将无法匹配的词放入unmatched_words，但不进行插入
                        unmatched_words.append(word)
                else:
                    # 已经在句子中
                    first_pass_words.append(Word(
                        text=word.text.strip(),
                        start=float(word.segment.start),
                        end=float(word.segment.end),
                        speaker=word.speaker,
                        language=word.language
                    ))
                    
                    # 检查是否形成整句结束（使用标点判断）
                    current_text = ' '.join(w.text for w in first_pass_words)
                    if (current_text.strip().endswith(('.', '?', '!', '。', '？', '！', '...', ')')) or 
                        current_segment.text.strip() in ['(Not clear)', '']):
                        sentence_started = False
                        # 不继续强行插入其他无法明确归属的词，全部进入unmatched备用
                    

            # 第一轮结束，此时first_pass_words中只包含了明确整句形成的词组
            current_segment.words = first_pass_words

            # 标记已分配的词
            for w in current_segment.words:
                w_key = f"{w.text}_{w.segment.start:.2f}_{w.segment.end:.2f}"
                assigned_words.add(w_key)

            # 将当前segment的结果存入临时结果
            # 如果没有形成整句，也无所谓，可能需要在第二轮处理
            first_round_results.append((current_segment, unmatched_words))

        # ========== 第二轮对齐（对空白处插入未匹配词） ==========
        # 现在 first_round_results 中有两类数据：
        # 1. current_segment.words: 已匹配的整句
        # 2. unmatched_words: 未匹配的词
        # 这些unmatched_words及对应的segment需要二次处理
        for (segment, unmatched_words) in first_round_results:
            # 构造只包含本segment的列表传给assign_word使用
            # 因为 assign_word 期望传入TranscriptSegment的列表
            segment_list = [segment]

            # 在此轮中，对unmatched_words使用三个规则尝试插入
            for word in unmatched_words:
                if word is None:
                    continue
                if self.assign_word(word, segment_list):
                    # 根据时间顺序插入到segment.words中
                    insert_index = 0
                    # 确保不会破坏第一轮锁定的句子结构：
                    # 你可以在这里添加更严格的条件，比如句子边界检查。
                    while (insert_index < len(segment.words) and 
                        segment.words[insert_index].segment.start < float(word.segment.start)):
                        insert_index += 1

                    # 插入前可以再次检查不破坏已经形成的整句（如is_breaking_sentence）
                    # 如果breaking，则continue跳过此词

                    segment.words.insert(insert_index, Word(
                        text=word.text.strip(),
                        start=float(word.segment.start),
                        end=float(word.segment.end),
                        speaker=word.speaker,
                        language=word.language
                        
                    ))
                    # 标记已分配的词
                    w_key = f"{word.text}_{word.segment.start:.2f}_{word.segment.end:.2f}"
                    assigned_words.add(w_key)

            # 生成最终的result字典
            result = {
                'human_onset': segment.onset,
                'human_timestamp': segment.get_original_timestamp(),
                'human_text': segment.text,
                'speaker': segment.speaker,
                'language': segment.language,
            }

            if segment.words:
                words_start = min(w.segment.start for w in segment.words)
                words_end = max(w.segment.end for w in segment.words)

                machine_speaker = segment.words[0].speaker
                machine_language = segment.words[0].language

                result.update({
                    'machine_onset': TimeSegment.format_time(words_start),
                    'machine_end': TimeSegment.format_time(words_end),
                    'machine_timestamp': f"{TimeSegment.format_time(words_start)} - {TimeSegment.format_time(words_end)}",
                    'machine_text': ' '.join(w.text for w in segment.words),
                    'word_count': len(segment.words),
                    'overlaps': any(
                        w.segment.start < segment.time_segment.start or 
                        w.segment.end > segment.time_segment.end 
                        for w in segment.words
                    ),
                    'machine_speaker': machine_speaker,
                    'machine_language': machine_language
                })
            else:
                result.update({
                    'machine_onset': '',
                    'machine_end': '',
                    'machine_timestamp': '',
                    'machine_text': '',
                    'word_count': 0,
                    'overlaps': False,
                    'machine_speaker': '',
                    'machine_language': ''
                })

            results.append(result)

        # 在最后对results进行排序
        results.sort(key=lambda x: convert_to_seconds(x['human_timestamp'].split(' - ')[0]))
        return results


    
    def _prepare_words(self, machine_data: List[Dict]) -> List[Word]:
        words = []
        for entry in machine_data:
            segment_lang = entry.get("Language", "")
            segment_speaker = entry.get("Speaker", "")
            for word_data in entry.get('Words', []):
                word = Word(
                    text=word_data['word'].strip(),
                    start=float(word_data['start']),
                    end=float(word_data['end']),
                    speaker=segment_speaker,
                    language=segment_lang
                )
                words.append(word)
        return words

    def _parse_timestamp(self, timestamp: str) -> Tuple[float, float]:
        try:
            start, end = timestamp.split(' - ')
            start_seconds = convert_to_seconds(start)
            end_seconds = convert_to_seconds(end)
            logger.debug(f"Parsing timestamp: {timestamp} -> {start_seconds}, {end_seconds}")
            return start_seconds, end_seconds
        except:
            logger.error(f"Failed to parse timestamp: {timestamp}")
            return 0.0, 0.0

def write_csv_output(results: List[Dict], output_file: str):
    if not results:
        return
        
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
    
    with open(output_file, 'w', encoding='utf-8', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--manual_file", required=True, help="人工转录txt文件路径")
    parser.add_argument("--machine_file", required=True, help="机器转录json文件路径")
    parser.add_argument("--output_csv", default="transcript_comparison.csv",
                        help="对齐后输出的初步CSV文件名，默认为 transcript_comparison.csv")
    args = parser.parse_args()
    #manual_file = 'v058.txt'
    #machine_file = 'Method4_small_WBW (3).json'
    #output_file = 'transcript_comparison_Method6.csv'
    
    try:
        #manual_data = load_manual_transcript(manual_file)
        manual_data = load_manual_transcript(args.manual_file)
        #machine_data = load_machine_transcript(machine_file)
        machine_data = load_machine_transcript(args.machine_file)
        
        aligner = TranscriptAligner()
        results = aligner.align(manual_data, machine_data)

        logger.info(f"Number of results: {len(results)}")
        for r in results[:5]:
            logger.info(r)
        
        #write_csv_output(results, output_file)
        write_csv_output(results, args.output_csv)
        
        logger.info(f"Successfully processed {len(results)} segments")
        #ogger.info(f"Results written to {output_file}")
        logger.info(f"Results written to {args.output_csv}")
        
    except Exception as e:
        logger.error(f"Error occurred: {str(e)}", exc_info=True)

if __name__ == "__main__":
    main()

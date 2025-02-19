import pandas as pd
import re
import argparse
from datetime import datetime
from opencc import OpenCC

def time_to_seconds(time_str):
    """
    能处理 HH:MM:SS, MM:SS, 以及 HH:MM:SS.ms 或 MM:SS.ms
    """
    try:
        if '.' in time_str:  # 带小数的格式
            main_part, ms_part = time_str.split('.')
            parts = main_part.split(':')
            # 根据 parts 长度判断是 HH:MM:SS 还是 MM:SS
            if len(parts) == 2:
                # MM:SS
                m, s = parts
                return float(m)*60 + float(s) + float(f'0.{ms_part}')
            elif len(parts) == 3:
                # HH:MM:SS
                h, m, s = parts
                return float(h)*3600 + float(m)*60 + float(s) + float(f'0.{ms_part}')
        else:
            # 不带小数点
            parts = time_str.split(':')
            if len(parts) == 2:
                return float(parts[0]) * 60 + float(parts[1])
            elif len(parts) == 3:
                return float(parts[0]) * 3600 + float(parts[1]) * 60 + float(parts[2])
        return 0
    except:
        return 0


def parse_timestamp_range(ts_str):
    """
    将 'HH:MM:SS - HH:MM:SS' 或 'MM:SS - MM:SS' 拆分成 (start_sec, end_sec)。
    如果解析失败，返回 (0, 0)。
    """
    try:
        start_str, end_str = ts_str.split(' - ')
        start_sec = time_to_seconds(start_str)
        end_sec = time_to_seconds(end_str)
        return start_sec, end_sec
    except:
        return 0, 0

def format_mmss(seconds):
    # 整体仍可先分分钟，再保留小数秒
    m = int(seconds // 60)
    s = seconds % 60  # s里保留小数
    return f"{m:02d}:{s:05.2f}"

def count_mixed_text(text):
    """
    统计文本中的【中文字符数】和【英文字数】。
    """
    if pd.isna(text):
        return 0, 0
    # 将文本分割成中文字符与非中文部分
    parts = re.findall(r'[\u4e00-\u9fff]|[^\u4e00-\u9fff]+', str(text))
    chinese_count = sum(1 for part in parts if re.match(r'[\u4e00-\u9fff]', part))
    english_count = sum(len(re.findall(r'\b[a-zA-Z]+\b', part)) 
                       for part in parts if not re.match(r'[\u4e00-\u9fff]', part))
    return chinese_count, english_count

def store_mixed_text(text):
    """
    统计文本中的【中文字符】和【英文字词】，并进行预处理。
    """
    text = normalize_text(text)
    if pd.isna(text):
        return set(), set()
    chinese_chars = set(re.findall(r'[一-鿿]', text))
    english_words = set(re.findall(r'\b[a-zA-Z]+\b', text))
    return chinese_chars, english_words

def normalize_text(text):
    """
    将文本中的繁体中文转换为简体中文，英文转换为小写。
    """
    if pd.isna(text):
        return ""
    cc = OpenCC('t2s')  # 繁体转简体
    text = cc.convert(text)
    return text.lower()

def compute_symmetric_difference(set1, set2):
    """
    计算对称差值方式的文本差异。
    """
    return max(len(set1), len(set2)) - len(set1 & set2)


def clean_text(text):
    """
    模拟 PyQt 中的 clean_text:
    只保留字母和数字，并转小写，忽略标点、空白等。
    """
    if pd.isna(text):
        return ''
    return ''.join(char.lower() for char in str(text) if char.isalnum())

def normalize_language(lang_str: str) -> str:
    """
    将语言字符串进行统一处理。
    例如:
    - "en-US", "en-GB" 等 → "en"
    - 其他情况例如 "Chinese" 就保持原样
    - 你也可以根据实际需要添加更多映射规则
    """
    if not lang_str:
        return ""

    # 全部转小写
    lang_str = lang_str.strip().lower()

    # 如果包含 "en-" 之类的前缀，就简化成 "en"
    if lang_str.startswith(("en-us","en-US")):
        return "en"

    # 根据需要，你也可以做更多映射，比如:
    if lang_str in ["zh-cn", "zh-hans", "chinese"]:
         return "zh"

    return lang_str

def calculate_differences(df):
    """
    对传入的 DataFrame, 逐行比较 'machine' 与 'human' 的各字段差异.
    """
    results = []

    prev_machine_end_time = 0 # Offset time for the previous row, starting with 0 

    total_text = 0
    total_difference = 0
    
    for idx, row in df.iterrows():
        # 1) Onset time difference (单点时间差)
        machine_time = time_to_seconds(row['machine_onset'])
        human_time = time_to_seconds(row['human_onset'])

        time_diff = abs(machine_time - human_time)
        
        # 2) TimeStamp difference (区间差): 拆分 'start-end', 计算长度差
        machine_start, machine_end = parse_timestamp_range(str(row['machine_timestamp']))
        human_start, human_end = parse_timestamp_range(str(row['human_timestamp']))

        if machine_time == 0 and human_time is not None:
            time_diff = abs(human_time - prev_machine_end_time) #if no machine time is found, recalculate time_diff use offset from previous chunk
        if machine_end != 0: #to avoid multiple empty machine
            prev_machine_end_time = machine_end #update offset time for the next chunk

        duration_machine = abs(machine_end - machine_start)
        duration_human = abs(human_end - human_start)
        time_stamp_diff = abs(duration_machine - duration_human)

        # 3) 文本差异
        # (a) 中英文字数差
        #machine_ch, machine_en = count_mixed_text(row['machine_text'])
        #human_ch, human_en = count_mixed_text(row['human_text'])
        #text_diff_count = abs(machine_ch - human_ch) + abs(machine_en - human_en)

        machine_ch_set, machine_en_set = store_mixed_text(row['machine_text'])
        human_ch_set, human_en_set = store_mixed_text(row['human_text'])
        
        word_differences = compute_symmetric_difference(machine_en_set, human_en_set)
        char_differences = compute_symmetric_difference(machine_ch_set, human_ch_set)
        text_diff_count = word_differences + char_differences

        total_text = total_text + len(human_ch_set) + len(human_en_set)
        total_difference += text_diff_count
        # difference = max(machine_text, human_text) - len(word in both sets), which refers to the change of characters/words to fix the text

        # (b) 清洗后文本是否相同
        cleaned_machine = clean_text(row['machine_text'])
        cleaned_human = clean_text(row['human_text'])
        text_different = (cleaned_machine != cleaned_human)  # 忽略标点和大小写

        # 如果想更严格对齐 PyQt, 一旦字数差>0 或清洗后文本不同, 都算 Text_difference=TRUE
        text_diff_label = 'TRUE' if (text_diff_count > 0 or text_different) else 'FALSE'
        
        # 4) 语言差异
        machine_lang_norm = normalize_language(str(row['machine_language']).strip())
        human_lang_norm   = normalize_language(str(row['language']).strip())
        lang_different    = (machine_lang_norm != human_lang_norm)
        
        # 5) 说话者差异
        speaker_different = (str(row['machine_speaker']).strip() != str(row['speaker']).strip())

        # 6) 将比较结果收集到字典
        results.append({
            # Machine
            'OnsetTime1': row['machine_onset'],
            'TimeStamp1': row['machine_timestamp'],
            'Text1': row['machine_text'],
            'Language1': row['machine_language'],
            'Speaker1': row['machine_speaker'],

            # Human
            'OnsetTime2': row['human_onset'],
            'TimeStamp2': row['human_timestamp'],
            'Text2': row['human_text'],
            'Language2': row['language'],
            'Speaker2': row['speaker'],

            # Differences
            'Absolute_onset_time_difference': format_mmss(time_diff),   # MM:SS
            'TimeStamp_difference': format_mmss(time_stamp_diff),       # 同样用 MM:SS 格式
            'Text_difference_word_count': text_diff_count,
            'Text_difference': text_diff_label,
            'Language_difference': 'TRUE' if lang_different else 'FALSE',
            'Speaker_difference': 'TRUE' if speaker_different else 'FALSE'
        })
    
    # 返回包含所有比较结果的 DataFrame
    print(total_difference / total_text)
    return pd.DataFrame(results)

def main():
    # 读取CSV文件 (根据需要修改路径)
    input_file = 'transcript_test_single_align_Method5_过滤.csv'
    df = pd.read_csv(input_file)
    #parser = argparse.ArgumentParser()
    #parser.add_argument("--input_csv", required=True,
    #                    help="final_align生成的 CSV 文件")
    #parser.add_argument("--audio_id", required=True,
    #                   help="音频文件ID")
    #parser.add_argument("--method_name", required=True,
    #                   help="方法名称")
    #parser.add_argument("--output_csv", default="diff.csv",
    #                    help="差异分析的输出文件")
    #args = parser.parse_args()

    #df = pd.read_csv(args.input_csv)

    # 计算差异
    diff_df = calculate_differences(df)
    
    # 导出结果
    #if args.output_csv:
    #    diff_df.to_csv(args.output_csv, index=False)
    #    print(f"Analysis completed. Results saved to {args.output_csv}")
    diff_df.to_csv("test_Method_Compute.csv", index=False)

    
if __name__ == "__main__":
    main()

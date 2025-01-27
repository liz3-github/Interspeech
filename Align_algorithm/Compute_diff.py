import pandas as pd
import re
import argparse
from datetime import datetime

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

def clean_text(text):
    """
    模拟 PyQt 中的 clean_text:
    只保留字母和数字，并转小写，忽略标点、空白等。
    """
    if pd.isna(text):
        return ''
    return ''.join(char.lower() for char in str(text) if char.isalnum())

def calculate_differences(df):
    """
    对传入的 DataFrame, 逐行比较 'machine' 与 'human' 的各字段差异.
    """
    results = []
    
    for idx, row in df.iterrows():
        # 1) Onset time difference (单点时间差)
        machine_time = time_to_seconds(row['machine_onset'])
        human_time = time_to_seconds(row['human_onset'])
        time_diff = abs(machine_time - human_time)
        
        # 2) TimeStamp difference (区间差): 拆分 'start-end', 计算长度差
        machine_start, machine_end = parse_timestamp_range(str(row['machine_timestamp']))
        human_start, human_end = parse_timestamp_range(str(row['human_timestamp']))
        duration_machine = abs(machine_end - machine_start)
        duration_human = abs(human_end - human_start)
        time_stamp_diff = abs(duration_machine - duration_human)

        # 3) 文本差异
        # (a) 中英文字数差
        machine_ch, machine_en = count_mixed_text(row['machine_text'])
        human_ch, human_en = count_mixed_text(row['human_text'])
        text_diff_count = abs(machine_ch - human_ch) + abs(machine_en - human_en)
        
        # (b) 清洗后文本是否相同
        cleaned_machine = clean_text(row['machine_text'])
        cleaned_human = clean_text(row['human_text'])
        text_different = (cleaned_machine != cleaned_human)  # 忽略标点和大小写

        # 如果想更严格对齐 PyQt, 一旦字数差>0 或清洗后文本不同, 都算 Text_difference=TRUE
        text_diff_label = 'TRUE' if (text_diff_count > 0 or text_different) else 'FALSE'
        
        # 4) 语言差异
        lang_different = (str(row['machine_language']).strip() != str(row['language']).strip())
        
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
    return pd.DataFrame(results)

def main():
    # 读取CSV文件 (根据需要修改路径)
    #input_file = 'transcript_final_align.csv'
    #df = pd.read_csv(input_file)
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_csv", required=True,
                        help="final_align生成的 CSV 文件")
    parser.add_argument("--audio_id", required=True,
                       help="音频文件ID")
    parser.add_argument("--method_name", required=True,
                       help="方法名称")
    parser.add_argument("--output_csv", default="diff.csv",
                        help="差异分析的输出文件")
    args = parser.parse_args()

    df = pd.read_csv(args.input_csv)

    # 计算差异
    diff_df = calculate_differences(df)
    
    # 导出结果
    if args.output_csv:
        diff_df.to_csv(args.output_csv, index=False)
        print(f"Analysis completed. Results saved to {args.output_csv}")

    
if __name__ == "__main__":
    main()

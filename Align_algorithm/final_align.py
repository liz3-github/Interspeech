import re
import pandas as pd
import numpy as np

def parse_hhmmss_to_seconds(ts_val):
    """
    将时间值 (ts_val) 转换为浮点数秒数:
    - 如果是 None / NaN，返回 None
    - 如果是 int/float，直接返回 float(ts_val)
    - 如果是字符串，则尝试解析 'HH:MM:SS.xxx'、'MM:SS.xx' 等格式
    """
    if pd.isna(ts_val) or ts_val is None:
        return None
    
    # 如果是数值，直接当作秒数
    if isinstance(ts_val, (int, float)):
        return float(ts_val)
    
    # 否则当作字符串处理
    ts_str = str(ts_val).strip()
    if not ts_str:
        return None

    # 简单正则: 允许 [HH:]MM:SS(.xxx)?
    pattern = r"(?:(\d+):)?(\d+):(\d+(?:\.\d+)?)"
    match = re.match(pattern, ts_str)
    if not match:
        raise ValueError(f"无法解析时间戳: {ts_str}")

    hour_str = match.group(1)
    minute_str = match.group(2)
    second_str = match.group(3)

    hours = int(hour_str) if hour_str else 0
    minutes = int(minute_str) if minute_str else 0
    seconds = float(second_str) if second_str else 0.0

    total_sec = hours * 3600 + minutes * 60 + seconds
    return total_sec


def combine_same_timestamp(entries):
    """
    将同一侧(机器/人工)里“开始时间”相同的多行合并为一行。
    例如，如果有两条记录 start_time 都是 04:04.00，就把它们合并。
    
    entries: [(start_sec, row_dict), ...]
    返回合并后的新列表，仍是 [(start_sec, merged_row_dict), ...]。
    """
    if not entries:
        return []

    # 按 start_sec 排序，None 放最后
    def sort_key(x):
        return x[0] if x[0] is not None else float('inf')

    entries = sorted(entries, key=sort_key)

    merged_entries = []
    current_time = entries[0][0]
    current_rows = [entries[0][1]]

    for i in range(1, len(entries)):
        t, rowdict = entries[i]
        if t == current_time:
            # 时间戳相同，合并在一起
            current_rows.append(rowdict)
        else:
            # 时间戳变了，先把之前的合并输出
            merged_row = merge_rows(current_time, current_rows)
            merged_entries.append((current_time, merged_row))
            # 开始新的
            current_time = t
            current_rows = [rowdict]

    # 处理最后一批
    merged_row = merge_rows(current_time, current_rows)
    merged_entries.append((current_time, merged_row))

    return merged_entries


def merge_rows(start_time, row_dicts):
    """
    把多个 row_dict 合并成一个新的 row_dict。
    例如可以把文本拼接，或保留第一个/最后一个，按需定制。
    这里演示：如遇到 machine_text/human_text，就用换行拼接，其他列只保留第一条的值。
    """
    if not row_dicts:
        return {}

    # 先复制第一条，作为基准
    base = row_dicts[0].copy()
    for other in row_dicts[1:]:
        # 针对特定列拼接，如 machine_text / human_text
        for k in ["machine_text", "human_text"]:
            if k in other and pd.notna(other[k]):
                base_txt = str(base.get(k, "")).strip()
                other_txt = str(other[k]).strip()
                if other_txt:
                    if base_txt:
                        base[k] = base_txt + "\n" + other_txt
                    else:
                        base[k] = other_txt
    return base


def align_transcripts(machine_entries, human_entries, threshold=1.0):
    """
    用双指针遍历对齐：
    - machine_entries: [(m_start_sec, m_rowdict), ...]
    - human_entries:   [(h_start_sec, h_rowdict), ...]
    - threshold:       “开始时间差值”阈值(秒)
    """
    # 先按开始时间排序；None 排到末尾
    def sort_key(x):
        return x[0] if x[0] is not None else float('inf')
    
    machine_entries = sorted(machine_entries, key=sort_key)
    human_entries   = sorted(human_entries,   key=sort_key)

    i, j = 0, 0
    aligned_rows = []

    while i < len(machine_entries) and j < len(human_entries):
        m_start_sec, m_row = machine_entries[i]
        h_start_sec, h_row = human_entries[j]

        # 对于 None 处理
        if m_start_sec is None and h_start_sec is None:
            combined = combine_machine_human(m_row, h_row)
            aligned_rows.append(combined)
            i += 1
            j += 1
            continue

        if m_start_sec is None:
            combined = combine_machine_human(m_row, None)
            aligned_rows.append(combined)
            i += 1
            continue
        
        if h_start_sec is None:
            combined = combine_machine_human(None, h_row)
            aligned_rows.append(combined)
            j += 1
            continue

        diff = abs(m_start_sec - h_start_sec)
        if diff <= threshold:
            # 时间接近，视为同一行
            combined = combine_machine_human(m_row, h_row)
            aligned_rows.append(combined)
            i += 1
            j += 1
        else:
            # 看谁开始时间更早
            if m_start_sec < h_start_sec:
                combined = combine_machine_human(m_row, None)
                aligned_rows.append(combined)
                i += 1
            else:
                combined = combine_machine_human(None, h_row)
                aligned_rows.append(combined)
                j += 1

    # 剩余机器条目
    while i < len(machine_entries):
        m_start_sec, m_row = machine_entries[i]
        combined = combine_machine_human(m_row, None)
        aligned_rows.append(combined)
        i += 1

    # 剩余人工条目
    while j < len(human_entries):
        h_start_sec, h_row = human_entries[j]
        combined = combine_machine_human(None, h_row)
        aligned_rows.append(combined)
        j += 1

    return aligned_rows


def combine_machine_human(m_dict, h_dict):
    """
    将机器的行信息 (m_dict) 和人工的行信息 (h_dict) 合并成一个新的 dict。
    如果某侧为 None, 则那侧所有列留空。

    已知原 CSV 的列有:
    - machine_onset, machine_end, machine_timestamp, machine_text
    - human_onset,   human_timestamp, human_text
    - speaker, language, word_count, overlaps, human_start_seconds

    """
    all_cols = [
        "machine_onset", "machine_end", "machine_timestamp", "machine_text",
        "machine_speaker", "machine_language",
        "human_onset", "human_timestamp", "human_text",
        "speaker", "language", "word_count", "overlaps", "human_start_seconds"
    ]
    new_row = {}
    
    # 如果某侧 None，就造一个空的 dict，避免后面索引错误
    if m_dict is None:
        m_dict = {}
    if h_dict is None:
        h_dict = {}

    # 先取出机器这边的 onset/end
    val_machine_onset = m_dict.get("machine_onset", "")
    val_machine_end   = m_dict.get("machine_end", "")

    # 先取出人工这边的 onset/end
    val_human_onset = h_dict.get("human_onset", "")
    val_human_end   = h_dict.get("human_end", "")



    # 写回 h_dict，以便后面统一取值
    h_dict["human_onset"] = val_human_onset
    h_dict["human_end"]   = val_human_end

    for col in all_cols:
        vm = m_dict.get(col, "")
        vh = h_dict.get(col, "")

        if col.startswith("machine_"):
            new_row[col] = vm if pd.notna(vm) else ""
        elif col.startswith("human_"):
            new_row[col] = vh if pd.notna(vh) else ""
        else:
            # 对于公共列: 如果机器那边有值就用, 否则用人工那边
            new_row[col] = vm if vm else vh

    return new_row


def main(input_csv, output_csv, threshold=1.0):
    df = pd.read_csv(input_csv)
    df.fillna("", inplace=True)

    # 分别构建 machine_entries 和 human_entries
    machine_entries = []
    human_entries   = []

    for i, row in df.iterrows():
        row_dict = row.to_dict()

        # 机器开始时间
        m_onset_val = row_dict.get("machine_onset", None)
        m_start_sec = parse_hhmmss_to_seconds(m_onset_val)
        # 若有机器时间，就加入 machine_entries
        if m_onset_val not in [None, "", np.nan]:
            machine_entries.append((m_start_sec, row_dict))

        # 人工开始时间
        h_onset_val = row_dict.get("human_onset", None)
        h_start_sec = parse_hhmmss_to_seconds(h_onset_val)
        # 若有人工时间，就加入 human_entries
        if h_onset_val not in [None, "", np.nan]:
            human_entries.append((h_start_sec, row_dict))

    # （图二需求）先把同一侧相同开始时间的多行合并
    machine_entries = combine_same_timestamp(machine_entries)
    human_entries   = combine_same_timestamp(human_entries)

    # 对齐
    aligned_list = align_transcripts(machine_entries, human_entries, threshold=threshold)

    # 转 DataFrame
    df_aligned = pd.DataFrame(aligned_list)

    condition = (df_aligned['machine_text'].str.strip() == "") & (df_aligned['human_text'].str.strip() == "")
    df_aligned = df_aligned[~condition].copy()
    # 输出到 CSV
    df_aligned.to_csv(output_csv, index=False)
    print(f"对齐完成，已输出到: {output_csv}")


if __name__ == "__main__":
    # 示例用法:
    input_file = "transcript_comparison_Method6.csv"
    output_file = "transcript_final_align.csv"
    time_threshold = 1.0  # 如果机器/人工的 start 时间差 <=1秒，则视为同一行

    main(input_file, output_file, threshold=time_threshold)

import os
import re
import math
import numpy as np
import pandas as pd
import datetime

# Time & Language Utilities

def parse_time_str(time_str):
    """
    Convert a time string (e.g., "00:07", "00:07:15", or "00.00") to seconds.
    If input is numeric, return its float value.
    """
    if time_str is None:
        return 0.0
    if isinstance(time_str, (int, float)):
        return float(time_str)
    time_str = time_str.strip()
    if ':' in time_str:
        parts = time_str.split(':')
        if len(parts) == 2:
            minutes, seconds = parts
            return int(minutes) * 60 + float(seconds)
        elif len(parts) == 3:
            hours, minutes, seconds = parts
            return int(hours) * 3600 + int(minutes) * 60 + float(seconds)
    elif '.' in time_str:
        try:
            return float(time_str)
        except ValueError:
            return 0.0
    else:
        try:
            return float(time_str)
        except ValueError:
            return 0.0

def seconds_to_hms(seconds):
    """
    Convert seconds to an "hh:mm:ss" string.
        NaN or None returns "00:00:00".
    """
    if seconds is None or (isinstance(seconds, float) and math.isnan(seconds)):
        seconds = 0
    seconds = int(round(seconds))
    h = seconds // 3600
    m = (seconds % 3600) // 60
    s = seconds % 60
    return f"{h:02d}:{m:02d}:{s:02d}"

def classify_lang(lang_str):
    """
    Normalize a language string into one of: "en", "zh", "mixed", or "other".
    If it contains both "en" and "zh" (ignoring case and spaces), return "mixed".
    """
    if not lang_str or pd.isna(lang_str) or lang_str.strip() == "":
        return "other"
    lang_str = lang_str.lower().replace(" ", "")
    if "en" in lang_str and "zh" in lang_str:
        return "mixed"
    elif "en" in lang_str:
        return "en"
    elif "zh" in lang_str:
        return "zh"
    else:
        return "other"

# Ground Truth Processing

def parse_txt_file(file_path):
    """
    Parse a TXT file with lines like:
      Timestamp: 00:07 - 00:51
      OnsetTime: 00:07
      Text: ...
      Speaker: ...
      Lang: en/zh/en+zh
    Returns a list of segments with start, end, duration, text, speaker,
    original language, and normalized category.
    """
    segments = []
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    pattern = r"Timestamp:\s*([0-9:.]+)\s*-\s*([0-9:.]+).*?Text:\s*(.*?)\s*Speaker:\s*(.*?)\s*Lang:\s*(.*?)(?=Timestamp:|$)"
    matches = re.findall(pattern, content, re.DOTALL)
    for match in matches:
        start_str, end_str, text, speaker, lang = match
        start = parse_time_str(start_str)
        end = parse_time_str(end_str)
        duration = end - start
        category = classify_lang(lang)
        segments.append({
            "start": start,
            "end": end,
            "duration": duration,
            "text": text.strip(),
            "speaker": speaker.strip(),
            "lang": lang.strip(),
            "category": category
        })
    return segments

def compute_gt_stats(gt_folder):
    """
    Process all TXT files in gt_folder to aggregate ground truth data.
    Returns a stats dictionary and a DataFrame with all segments.
    """
    gt_files = [os.path.join(gt_folder, f) for f in os.listdir(gt_folder) if f.endswith('.txt')]
    all_segments = []
    for f in gt_files:
        segments = parse_txt_file(f)
        all_segments.extend(segments)
    df = pd.DataFrame(all_segments)
    total_duration = df['duration'].sum() # in seconds
    num_segments = len(df)
    lang_distribution = df['lang'].value_counts().to_dict()  # original language count
    mean_duration = df['duration'].mean() if num_segments > 0 else 0
    median_duration = df['duration'].median() if num_segments > 0 else 0
    stats = {
        "total_duration": total_duration,
        "num_segments": num_segments,
        "lang_distribution": lang_distribution,
        "mean_segment_duration": mean_duration,
        "median_segment_duration": median_duration
    }
    return stats, df

def compute_language_stats(df):
    """
    Compute stats by language category ("en", "zh", "mixed", "other"):
      - Total duration (sec)
      - Segment count
      - Mean and median duration (ms)
    Returns a dictionary.
    """
    categories = ["en", "zh", "mixed", "other"]
    lang_stats = {}
    for cat in categories:
        df_cat = df[df["category"] == cat]
        total_duration_sec = df_cat["duration"].sum()
        count = len(df_cat)
        mean_duration_ms = df_cat["duration"].mean() * 1000 if count > 0 else 0
        median_duration_ms = df_cat["duration"].median() * 1000 if count > 0 else 0
        lang_stats[cat] = {
            "total_duration_sec": total_duration_sec,
            "segment_count": count,
            "mean_duration_ms": mean_duration_ms,
            "median_duration_ms": median_duration_ms
        }
    return lang_stats

def generate_table1(gt_stats, lang_stats, num_recordings):
    """
    Generate Table 1: an overview of ground truth data.
    Durations are formatted as hh:mm:ss.
    """
    table1 = {
        "Metric": [
            "#Recordings",
            "Total Duration (hh:mm:ss)",
            "#Segments",
            "English Duration (hh:mm:ss)",
            "Mandarin Duration (hh:mm:ss)",
            "Mixed Duration (hh:mm:ss)",
            "Other Duration (hh:mm:ss)",
            "#English Segments",
            "#Mandarin Segments",
            "#Mixed Segments",
            "#Other Segments",
            "Median Eng Segment (ms)",
            "Median Man Segment (ms)",
            "Median Mixed Segment (ms)",
            "Median Other Segment (ms)",
            "Mean Eng Segment (ms)",
            "Mean Man Segment (ms)",
            "Mean Mixed Segment (ms)",
            "Mean Other Segment (ms)"
        ],
        "Value": [
            num_recordings,
            seconds_to_hms(gt_stats["total_duration"]),
            gt_stats["num_segments"],
            seconds_to_hms(lang_stats["en"]["total_duration_sec"]),
            seconds_to_hms(lang_stats["zh"]["total_duration_sec"]),
            seconds_to_hms(lang_stats["mixed"]["total_duration_sec"]),
            seconds_to_hms(lang_stats["other"]["total_duration_sec"]),
            lang_stats["en"]["segment_count"],
            lang_stats["zh"]["segment_count"],
            lang_stats["mixed"]["segment_count"],
            lang_stats["other"]["segment_count"],
            round(lang_stats["en"]["median_duration_ms"]),
            round(lang_stats["zh"]["median_duration_ms"]),
            round(lang_stats["mixed"]["median_duration_ms"]),
            round(lang_stats["other"]["median_duration_ms"]),
            round(lang_stats["en"]["mean_duration_ms"]),
            round(lang_stats["zh"]["mean_duration_ms"]),
            round(lang_stats["mixed"]["mean_duration_ms"]),
            round(lang_stats["other"]["mean_duration_ms"])
        ]
    }
    df_table1 = pd.DataFrame(table1)
    return df_table1

# Auto-alignment Data Processing

def compute_csv_stats(csv_file):
    """
    Parse an auto-alignment CSV file with columns:
    machine_onset, machine_end, machine_timestamp, machine_text, machine_speaker, machine_language, ...
    Compute segment durations, total duration, segment count, and per-language durations.
    """
    df = pd.read_csv(csv_file)
    df['start'] = df['machine_onset'].apply(parse_time_str)
    df['end'] = df['machine_end'].apply(parse_time_str)
    df['duration'] = df['end'] - df['start']
    total_duration = df['duration'].sum()
    num_segments = len(df)
    df['category'] = df['machine_language'].apply(lambda x: classify_lang(x))
    lang_counts = df['category'].value_counts().to_dict()
    mean_duration = df['duration'].mean() if num_segments > 0 else 0
    median_duration = df['duration'].median() if num_segments > 0 else 0
    stats = {
        "total_duration": total_duration,
        "num_segments": num_segments,
        "lang_distribution": lang_counts,
        "mean_segment_duration": mean_duration,
        "median_segment_duration": median_duration
    }
    return stats, df

def compute_variant_stats(method_folder, model_type):
    """
    For a given method's model folder (variant), read all CSV files and compute:
      - Total duration, segment count, overall mean and median duration (sec)
      - For each language category (en, zh, mixed, other): total duration, mean, median, and count
    Returns a stats dictionary.
    """
    csv_folder = os.path.join(method_folder, model_type)
    if not os.path.exists(csv_folder):
        alternative_folder = os.path.join(method_folder, "Model_Azure")
        if os.path.exists(alternative_folder):
            csv_folder = alternative_folder
        else:
            print(f"Warning: Neither {csv_folder} nor {alternative_folder} exist.")
            return None
    csv_files = [os.path.join(csv_folder, f) for f in os.listdir(csv_folder) if f.endswith('.csv')]
    if not csv_files:
        print(f"Warning: No CSV files found in {csv_folder}")
        return None
    df_list = []
    for file in csv_files:
        try:
            df = pd.read_csv(file)
            df['start'] = df['machine_onset'].apply(parse_time_str)
            df['end'] = df['machine_end'].apply(parse_time_str)
            df['duration'] = df['end'] - df['start']
            df['category'] = df['machine_language'].apply(lambda x: classify_lang(x))
            df_list.append(df)
        except Exception as e:
            print(f"Error processing {file}: {e}")
    if not df_list:
        return None
    df_all = pd.concat(df_list, ignore_index=True)
    total_duration = df_all['duration'].sum()
    num_segments = len(df_all)
    mean_duration = df_all['duration'].mean() if num_segments > 0 else 0
    median_duration = df_all['duration'].median() if num_segments > 0 else 0
    languages = ["en", "zh", "mixed", "other"]
    lang_stats = {}
    for lang in languages:
        df_lang = df_all[df_all["category"] == lang]
        count = len(df_lang)
        total_dur = df_lang['duration'].sum()
        mean_dur = df_lang['duration'].mean() if count > 0 else 0
        median_dur = df_lang['duration'].median() if count > 0 else 0
        lang_stats[lang] = {
            "total_duration": total_dur,
            "mean_duration": mean_dur,
            "median_duration": median_dur,
            "segment_count": count
        }
    return {
        "total_duration": total_duration,
        "num_segments": num_segments,
        "mean_duration": mean_duration,
        "median_duration": median_duration,
        "lang": lang_stats
    }

def combine_method_results(method_folder, model_list):
    """
    For a method with multiple models (e.g., small and medium), combine results by:
      - Summing total duration, segment counts, and per-language durations.
      - Computing overall mean and median durations from merged segments.
    Returns a combined stats dictionary.
    """
    results_list = []
    for model in model_list:
        res = compute_variant_stats(method_folder, model)
        if res is not None:
            results_list.append(res)
    if not results_list:
        return None
    combined_total_duration = sum(r["total_duration"] for r in results_list)
    combined_total_segments = sum(r["num_segments"] for r in results_list)
    all_durations = []
    combined_lang_duration = {}
    for r in results_list:
        all_durations.extend(r.get("all_durations", []))  # If there are separate all_durations
        for lang, stat in r["lang"].items():
            combined_lang_duration[lang] = combined_lang_duration.get(lang, 0) + stat["total_duration"]
    # If no all_durations information is available, the mean weighting can be combined
    all_durations = []
    for r in results_list:
        df_mean = r["mean_duration"]
        
        all_durations.append(r["mean_duration"])
    overall_mean = np.nanmean([r["mean_duration"] for r in results_list]) if results_list else 0
    overall_median = np.nanmedian([r["median_duration"] for r in results_list]) if results_list else 0
    return {
        "total_duration": combined_total_duration,
        "num_segments": combined_total_segments,
        "lang_duration": combined_lang_duration,
        "mean_duration": overall_mean,
        "median_duration": overall_median
    }

def generate_table2_pivot_variant(methods_list, gt_stats):
    """
    Generate pivot Table 2 with columns for each method variant (e.g., "Method1 small", "Method1 medium", ...,
    "Method6 Azure") and rows:
      - Total Duration (hh:mm:ss)
      - #Segments
      - Duration vs GT (%)
      - Segments vs GT (%)
      - English Duration (hh:mm:ss)
      - Mean English Duration (ms)
      - Median English Duration (ms)
      - Mandarin Duration (hh:mm:ss)
      - Mean Mandarin Duration (ms)
      - Median Mandarin Duration (ms)
      - Mixed Duration (hh:mm:ss)
      - Mean Mixed Duration (ms)
      - Median Mixed Duration (ms)
      - Other Duration (hh:mm:ss)
      - Mean Other Duration (ms)
      - Median Other Duration (ms)
    """
    row_labels = [
        "Total Duration (hh:mm:ss)",
        "#Segments",
        "Duration vs GT (%)",
        "Segments vs GT (%)",
        "English Duration (hh:mm:ss)",
        "Mean English Duration (ms)",
        "Median English Duration (ms)",
        "Mandarin Duration (hh:mm:ss)",
        "Mean Mandarin Duration (ms)",
        "Median Mandarin Duration (ms)",
        "Mixed Duration (hh:mm:ss)",
        "Mean Mixed Duration (ms)",
        "Median Mixed Duration (ms)",
        "Other Duration (hh:mm:ss)",
        "Mean Other Duration (ms)",
        "Median Other Duration (ms)"
    ]
    pivot_data = {label: {} for label in row_labels}
    for method in methods_list:
        if method == "Method6":
            model_list = ["Model_Azure"]
        else:
            model_list = ["small", "medium"]
        for model in model_list:
            variant_name = f"{method} {model}" if model != "Model_Azure" else f"{method} Azure"
            method_folder = os.path.join("WBW_Align_new_Results", method)
            variant_stats = compute_variant_stats(method_folder, model)
            if variant_stats is None:
                for label in row_labels:
                    if "hh:mm:ss" in label:
                        pivot_data[label][variant_name] = "00:00:00"
                    else:
                        pivot_data[label][variant_name] = 0
                continue
            
            dur_vs_gt = ((variant_stats["total_duration"] - gt_stats["total_duration"]) / gt_stats["total_duration"] * 100) if gt_stats["total_duration"] else 0
            seg_vs_gt = ((variant_stats["num_segments"] - gt_stats["num_segments"]) / gt_stats["num_segments"] * 100) if gt_stats["num_segments"] else 0
            
            pivot_data["Total Duration (hh:mm:ss)"][variant_name] = seconds_to_hms(variant_stats["total_duration"])
            pivot_data["#Segments"][variant_name] = variant_stats["num_segments"]
            pivot_data["Duration vs GT (%)"][variant_name] = round(dur_vs_gt, 2)
            pivot_data["Segments vs GT (%)"][variant_name] = round(seg_vs_gt, 2)
            
            # English stats
            en_stats = variant_stats["lang"].get("en", {"total_duration": 0, "mean_duration": 0, "median_duration": 0})
            pivot_data["English Duration (hh:mm:ss)"][variant_name] = seconds_to_hms(en_stats["total_duration"])
            pivot_data["Mean English Duration (ms)"][variant_name] = round(en_stats["mean_duration"] * 1000) if en_stats["segment_count"] != 0 else 0
            pivot_data["Median English Duration (ms)"][variant_name] = round(en_stats["median_duration"] * 1000) if en_stats["segment_count"] != 0 else 0
            
            # Mandarin stats
            zh_stats = variant_stats["lang"].get("zh", {"total_duration": 0, "mean_duration": 0, "median_duration": 0})
            pivot_data["Mandarin Duration (hh:mm:ss)"][variant_name] = seconds_to_hms(zh_stats["total_duration"])
            pivot_data["Mean Mandarin Duration (ms)"][variant_name] = round(zh_stats["mean_duration"] * 1000) if zh_stats["segment_count"] != 0 else 0
            pivot_data["Median Mandarin Duration (ms)"][variant_name] = round(zh_stats["median_duration"] * 1000) if zh_stats["segment_count"] != 0 else 0
            
            # Mixed stats
            mix_stats = variant_stats["lang"].get("mixed", {"total_duration": 0, "mean_duration": 0, "median_duration": 0})
            pivot_data["Mixed Duration (hh:mm:ss)"][variant_name] = seconds_to_hms(mix_stats["total_duration"])
            pivot_data["Mean Mixed Duration (ms)"][variant_name] = round(mix_stats["mean_duration"] * 1000) if mix_stats["segment_count"] != 0 else 0
            pivot_data["Median Mixed Duration (ms)"][variant_name] = round(mix_stats["median_duration"] * 1000) if mix_stats["segment_count"] != 0 else 0
            
            # Other stats
            other_stats = variant_stats["lang"].get("other", {"total_duration": 0, "mean_duration": 0, "median_duration": 0, "segment_count": 0})
            pivot_data["Other Duration (hh:mm:ss)"][variant_name] = seconds_to_hms(other_stats["total_duration"])
            other_mean = np.nan_to_num(other_stats.get("mean_duration", 0))
            other_median = np.nan_to_num(other_stats.get("median_duration", 0))
            pivot_data["Mean Other Duration (ms)"][variant_name] = round(other_mean * 1000)
            pivot_data["Median Other Duration (ms)"][variant_name] = round(other_median * 1000)
            
    df_pivot = pd.DataFrame(pivot_data)
    df_pivot.index.name = "Metric"
    df_pivot = df_pivot.transpose()
    return df_pivot

# Main Process

def main():
    # Ground Truth
    gt_folder = "annotated_merged"  # Modify path if needed
    gt_stats, gt_df = compute_gt_stats(gt_folder)
    num_recordings = len([f for f in os.listdir(gt_folder) if f.endswith('.txt')])
    lang_stats = compute_language_stats(gt_df)
    df_table1 = generate_table1(gt_stats, lang_stats, num_recordings)
    print("Table 1 (Ground Truth Overview):")
    print(df_table1)
    df_table1.to_csv("table1_ground_truth.csv", index=False)
    with open("table1.md", "w", encoding="utf-8") as f:
        f.write(df_table1.to_markdown())
    
    # Auto-alignment results: Generate Pivot Table
    methods_list = ["Method1", "Method2", "Method3", "Method4", "Method5", "Method6"]
    df_table2 = generate_table2_pivot_variant(methods_list, gt_stats)
    print("Table 2 (Pivot):")
    print(df_table2)
    df_table2.to_csv("table2_comparison_pivot.csv", index=True)
    with open("table2.md", "w", encoding="utf-8") as f:
        f.write(df_table2.to_markdown())

if __name__ == "__main__":
    main()

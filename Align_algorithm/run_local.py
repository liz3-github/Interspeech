import os
import subprocess
import glob
from tqdm import tqdm

# 基础路径配置
BASE_DIR = os.path.dirname(os.path.abspath(__file__))  # 获取当前文件所在目录
TXT_DIR = os.path.join(BASE_DIR, "annotated")  # 人工标注文件目录
WBW_MODEL_DIR = os.path.join(BASE_DIR, "WBW_Run_Model")  # 模型运行结果目录
ALIGN_SCRIPT_DIR = os.path.join(BASE_DIR, "Align_algorithm")  # 对齐算法脚本目录
OUTPUT_DIR = os.path.join(BASE_DIR, "WBW_Align_Results")  # 输出结果目录

# 创建输出目录
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 指定处理方法
METHODS = ["Method6"]

def get_audio_folder_names():
    """返回 WBW_MODEL_DIR 下所有子文件夹名称"""
    if not os.path.exists(WBW_MODEL_DIR):
        print(f"Warning: Directory not found: {WBW_MODEL_DIR}")
        return []
    
    all_subdirs = []
    for entry in os.scandir(WBW_MODEL_DIR):
        if entry.is_dir():
            all_subdirs.append(entry.name)
    return sorted(all_subdirs)

def get_manual_txt_names():
    """返回 TXT_DIR 下所有 .txt 文件名"""
    if not os.path.exists(TXT_DIR):
        print(f"Warning: Directory not found: {TXT_DIR}")
        return []
    
    txt_paths = glob.glob(os.path.join(TXT_DIR, "*.txt"))
    txt_names_no_ext = [os.path.splitext(os.path.basename(p))[0] for p in txt_paths]
    return sorted(txt_names_no_ext)

def run_pipeline(audio_id, method, manual_file, machine_file, model_size):
    """处理单个文件的pipeline"""
    print(f"\nProcessing: {audio_id}")
    print(f"Manual file: {manual_file}")
    print(f"Machine file: {machine_file}")
    
    # 创建输出目录
    method_size_dir = os.path.join(OUTPUT_DIR, method)
    os.makedirs(method_size_dir, exist_ok=True)

    # 检查文件
    if not os.path.exists(manual_file) or not os.path.exists(machine_file):
        print(f"[ERROR] Input file not found")
        return False
    
    if os.path.getsize(manual_file) == 0 or os.path.getsize(machine_file) == 0:
        print(f"[WARN] Input file is empty, skip.")
        return False

    # 设置输出文件路径
    init_csv = os.path.join(method_size_dir, f"{audio_id}_init.csv")
    final_csv = os.path.join(method_size_dir, f"{audio_id}_final.csv")
    ratio_txt = os.path.join(method_size_dir, f"{audio_id}_ratio.txt")
    diff_csv = os.path.join(method_size_dir, f"{audio_id}_diff.csv")

    # 执行各个步骤
    commands = [
        ["python", os.path.join(ALIGN_SCRIPT_DIR, "Intial_Align.py"),
         "--manual_file", manual_file, "--machine_file", machine_file, "--output_csv", init_csv],
        ["python", os.path.join(ALIGN_SCRIPT_DIR, "final_align.py"),
         "--input_csv", init_csv, "--output_csv", final_csv, "--threshold", "1.0"],
        ["python", os.path.join(ALIGN_SCRIPT_DIR, "align_word_ratio.py"),
         "--machine_json", machine_file, "--aligned_csv", final_csv, "--output_txt", ratio_txt],
        ["python", os.path.join(ALIGN_SCRIPT_DIR, "Compute_diff.py"),
         "--input_csv", final_csv, "--audio_id", audio_id, "--method_name", method, "--output_csv", diff_csv]
    ]

    for cmd in commands:
        print("Running:", " ".join(cmd))
        try:
            ret = subprocess.run(cmd, capture_output=True, text=True)
            if ret.returncode != 0:
                print(f"[ERROR] Command failed with code {ret.returncode}")
                print("STDOUT:", ret.stdout)
                print("STDERR:", ret.stderr)
                return False
        except Exception as e:
            print(f"[ERROR] Failed to run command: {e}")
            return False

    print(f"[INFO] Done pipeline for {audio_id} - {method}")
    return True

def main():
    # 1) 收集并匹配文件
    print("\nCollecting files...")
    audio_folders = get_audio_folder_names()
    txt_names = get_manual_txt_names()
    
    if not audio_folders or not txt_names:
        print("No files found to process!")
        return
    
    audio_set = set(audio_folders)
    txt_set = set(txt_names)
    matched = sorted(audio_set.intersection(txt_set))

    if not matched:
        print("No matching files found!")
        return

    print(f"\n==== Found {len(matched)} matching files")
    
    # 可以限制处理文件数量进行测试
    # matched = matched[:2]  # 只处理前2个文件
    
    # 处理所有文件
    print("\nProcessing files...")
    processed_count = 0
    for folder_name in tqdm(matched):
        manual_file = os.path.join(TXT_DIR, folder_name + ".txt")
        
        for method in METHODS:
            json_path = os.path.join(WBW_MODEL_DIR, folder_name, method)
            if os.path.exists(json_path):
                json_pattern = f"{method}_Model_Azure_WBW.json"
                json_file = os.path.join(json_path, json_pattern)
                
                if os.path.exists(json_file):
                    print(f"\n=== Processing {folder_name} / {method} / {os.path.basename(json_file)} ===")
                    if run_pipeline(folder_name, method, manual_file, json_file, ""):
                        processed_count += 1

    print(f"\nSuccessfully processed {processed_count} out of {len(matched)} files")
    print("All done!")

if __name__ == "__main__":
    main()

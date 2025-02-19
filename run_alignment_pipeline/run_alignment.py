import os
import glob
import subprocess
from tqdm import tqdm

#===== Base paths configuration =====
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

TXT_DIR = os.path.join(BASE_DIR, "annotated_merged")        # Directory of manual annotation files
WBW_MODEL_DIR = os.path.join(BASE_DIR, "WBW_Run_Model_New(2_4)")  # Directory of model run results
ALIGN_SCRIPT_DIR = os.path.join(BASE_DIR, "Align_algorithm") # Directory of alignment scripts
OUTPUT_DIR = os.path.join(BASE_DIR, "WBW_Align_Results")  # Directory for output results

os.makedirs(OUTPUT_DIR, exist_ok=True)

# Adjustable list: Methods and Models to process
METHODS = ["Method1", "Method2", "Method3", "Method4", "Method5", "Method6"]
MODELS  = ["small", "medium", "azure"]  


def get_audio_folder_names():
    """Return all subfolder names in WBW_MODEL_DIR (audio IDs)."""
    if not os.path.exists(WBW_MODEL_DIR):
        print(f"Warning: Directory not found: {WBW_MODEL_DIR}")
        return []
    all_subdirs = []
    for entry in os.scandir(WBW_MODEL_DIR):
        if entry.is_dir():
            all_subdirs.append(entry.name)
    return sorted(all_subdirs)

def get_manual_txt_names():
    """Return all .txt file names (without extension) in TXT_DIR."""
    if not os.path.exists(TXT_DIR):
        print(f"Warning: Directory not found: {TXT_DIR}")
        return []
    txt_paths = glob.glob(os.path.join(TXT_DIR, "*.txt"))
    txt_names_no_ext = [os.path.splitext(os.path.basename(p))[0] for p in txt_paths]
    return sorted(txt_names_no_ext)

def run_pipeline(audio_id, method, manual_file, machine_file, model_size):
    """
    Run new_align.py and align_word_ratio.py for a single audio ID:
      1) new_align: perform alignment
      2) align_word_ratio: calculate ratio
    """
    print(f"\n[INFO] Processing: audio_id={audio_id}, method={method}, model={model_size}")
    print(f"       manual_file={manual_file}")
    print(f"       machine_file={machine_file}")

    # Create output directory (e.g., OUTPUT_DIR/Method3/small/)
    method_size_dir = os.path.join(OUTPUT_DIR, method, model_size)
    os.makedirs(method_size_dir, exist_ok=True)

    # Basic check for input files
    if not os.path.exists(manual_file) or not os.path.exists(machine_file):
        print(f"[ERROR] Input file not found, skip.")
        return False
    if os.path.getsize(manual_file) == 0 or os.path.getsize(machine_file) == 0:
        print(f"[WARN] Input file is empty, skip.")
        return False

    # Set output file paths
    final_csv = os.path.join(method_size_dir, f"{audio_id}_final.csv")
    ratio_txt = os.path.join(method_size_dir, f"{audio_id}_ratio.txt")

    # call new_align and align_word_ratio scripts
    commands = [
        [
            "python",
            os.path.join(ALIGN_SCRIPT_DIR, "align_algorithm.py"),
            "--manual_file", manual_file,
            "--machine_file", machine_file,
            "--output_csv", final_csv
        ],
        [
            "python",
            os.path.join(ALIGN_SCRIPT_DIR, "align_word_ratio.py"),
            "--machine_json", machine_file,
            "--aligned_csv", final_csv,
            "--output_txt", ratio_txt
        ]
    ]

    for cmd in commands:
        cmd_str = " ".join(cmd)
        print(f"Running: {cmd_str}")
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

    print(f"[INFO] Done pipeline for audio_id={audio_id}, method={method}, model={model_size}")
    return True

def main():
    print("[INFO] Collecting audio folders and manual txt files...")

    audio_folders = get_audio_folder_names()
    txt_names = get_manual_txt_names()

    # Find the intersection of audio IDs and manual TXT names
    matched = sorted(set(audio_folders).intersection(set(txt_names)))
    print(f"[INFO] Found {len(matched)} matching IDs.\n")

    processed_count = 0

    for folder_name in tqdm(matched):
        manual_file = os.path.join(TXT_DIR, folder_name + ".txt")

        # Iterate over required Methods
        for method in METHODS:
            method_dir = os.path.join(WBW_MODEL_DIR, folder_name, method)
            if not os.path.isdir(method_dir):
                # Skip if the method subfolder doesn't exist
                continue

            # Iterate over desired Models
            for model_size in MODELS:
                if model_size == "azure" and method == "Method6":
                    # Method6 + azure 
                    json_pattern = f"{method}_Model_Azure_WBW.json"
                else:
                    ## General pattern
                    json_pattern = f"{method}_{model_size}_WBW.json"

                json_file = os.path.join(method_dir, json_pattern)
                if not os.path.exists(json_file):
                    # Skip if no matching JSON is found
                    continue

                # Run the alignment pipeline
                ok = run_pipeline(folder_name, method, manual_file, json_file, model_size)
                if ok:
                    processed_count += 1

    print(f"\n[INFO] Successfully processed {processed_count} items!")
    print("[INFO] All done.")

if __name__ == "__main__":
    main()

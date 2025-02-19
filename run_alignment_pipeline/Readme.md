# ASR Preprocessing & Alignment Pipeline

This folder (**Preprocessing_and_alignment/**) contains all scripts necessary for:

1. **Preprocessing** raw human transcripts (**annotated/** → **annotated_merged/**)
2. **Aligning** human transcripts with machine outputs
3. **Computing** ratio/word-level statistics
4. **Performing** a final overall analysis (generating pivot tables, comparing to ground truth)

## Folder Structure
```
Preprocessing_and_alignment/
    ├── align_algorithm.py             # Core alignment logic (manual vs. machine transcripts)
    ├── alignment_ratio.py             # Calculates unmatched/matched word ratio
    ├── preprocess_GroundTruth.py      # Cleans & merges raw .txt transcripts -> annotated_merged/GT
├── run_alignment.py               # Main execution script to align each method/model
├── analysis_overall.py            # Generates final tables (table1, table2) comparing all methods/models to GT
├── table1_ground_truth.csv        # Example output: ground truth stats
├── table2_comparison_pivot.csv    # Example output: pivot table comparing different methods
└── README.md                      # Documentation of pipeline usage
```
---

## 1. Preprocessing: `preprocess_GroundTruth.py`

### Purpose:
Reads the **annotated/** folder’s raw transcripts, removes empty segments, merges adjacent timestamps, and writes cleaned `.txt` files into **annotated_merged/**.

### Usage:
```sh
python preprocess_GroundTruth.py
```
Ensure that **annotated/** contains your original `.txt` transcripts. After running, **annotated_merged/** will hold the processed files.

---

## 2. Alignment: `align_algorithm.py` & `run_alignment.py`

### Core Script: `align_algorithm.py`
- Performs a two-round alignment between manual `.txt` (in **annotated_merged/**) and machine `.json` (in **WBW_Run_Model_New(2_4)/MethodX/ModelY**), outputting a `.csv` with matched text segments.

### Driver Script: `run_alignment.py`
- Iterates over each method and model, finds matching `.txt + .json`, calls `align_algorithm.py` internally, and writes results to **WBW_Align_Results/Method/Model/ID_final.csv**.

### Usage:
```sh
python run_alignment.py
```
This assumes you’ve already run `preprocess_GroundTruth.py` and have the machine JSON files in the correct folders.

---

## 3. Ratio Computation: `alignment_ratio.py`

### Purpose:
- Given a machine transcript JSON and the final aligned CSV, calculates how many words remain unmatched using:
  ```
  (total_words - matched_words) / total_words * 100
  ```

### Typical Usage:
Called within or after the alignment step. You can also run it manually:
```sh
python alignment_ratio.py \
    --machine_json path/to/MethodX_ModelY.json \
    --aligned_csv path/to/<ID>_final.csv \
    --output_txt path/to/<ID>_ratio.txt
```
Writes the ratio information to the console and/or a text file.

---

## 4. Overall Analysis: `analysis_overall.py`

### Purpose:
- Computes ground truth stats (**table1**) by scanning **annotated_merged/** for durations, segments, and language distributions.
- Aggregates all methods’ alignment results in **WBW_Align_new_Results/**, producing a pivot table (**table2**) comparing each method-model pair to the ground truth.

### Outputs:
- **table1_ground_truth.csv** / **table1.md**: Ground truth overall stats (#Segments, total duration, etc.).
- **table2_comparison_pivot.csv** / **table2.md**: Pivot table listing total duration, #segments, language breakdown, plus **“Duration vs GT (%)”** for each method-model variant.

### Usage:
```sh
python analysis_overall.py
```
Make sure you already have results from your alignment step in **WBW_Align_new_Results/MethodX/...**.

---

## Example Workflow

### 1. Preprocess:
```sh
python preprocess_GroundTruth.py
```

### 2. Align:
```sh
python run_alignment.py
```
This will call `align_algorithm.py` for each audio ID, method, and model, and optionally `alignment_ratio.py`.

### 3. Analyze:
```sh
python analysis_overall.py
```
Outputs **table1_ground_truth.csv/md** and **table2_comparison_pivot.csv/md** in this directory (or wherever you specify).

---

## Notes
- **table1_ground_truth.csv** and **table2_comparison_pivot.csv** seen here are example outputs after running `analysis_overall.py`.
- If you want to keep the results separate, feel free to move them into a **results/** subfolder or store them in **WBW_Align_Results/**.

---

## Troubleshooting

### **1. Directory Not Found**
- Check that your folders **annotated_merged/**, **WBW_Run_Model_New(2_4)**, and **WBW_Align_Results/** exist.

### **2. Empty or Missing `.txt/.json`**
- Scripts will skip missing files and print a warning.

### **3. No `tableX` output**
- Confirm that alignment actually ran and you have `.csv` files to analyze.

---

This document provides an overview of the **ASR Preprocessing & Alignment Pipeline**. If you have further questions, refer to the script comments or reach out for additional support.


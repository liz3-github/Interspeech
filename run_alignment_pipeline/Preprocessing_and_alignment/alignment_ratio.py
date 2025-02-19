import json
import csv
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--machine_json", required=True,
                        help="Machine transcript JSON file")
    parser.add_argument("--aligned_csv", required=True,
                        help="CSV output from final_align")
    parser.add_argument("--output_txt", default="",
                        help="Optional: write ratio info to a txt file")
    args = parser.parse_args()

    # Load machine transcript JSON
    with open(args.machine_json, 'r', encoding='utf-8') as f:
        word_by_word = json.load(f)

    # Count total words in the JSON (each segment's 'Words' field)
    total_words = sum(len(segment['Words']) for segment in word_by_word if 'Words' in segment)

    # Load CSV and count words in the 'machine_text' field
    words_in_aligned = 0
    with open(args.aligned_csv, 'r', encoding='utf-8') as f:
        csv_reader = csv.DictReader(f)
        for row in csv_reader:
            if row['machine_text'] and row['machine_text'].strip():
                words = row['machine_text'].split()
                words_in_aligned += len(words)

    ratio = (total_words - words_in_aligned) / total_words * 100
    # ratio2 = (words_in_aligned / total_words)*100

    result_str = (f"JSON文件中的总单词数: {total_words}\n"
                  f"CSV中机器转录文本的单词数: {words_in_aligned}\n"
                  f"未对齐比率: {ratio:.2f}%")

    print(result_str)

    # Write result to a text file if an output path is provided
    if args.output_txt:
        with open(args.output_txt, 'w', encoding='utf-8') as outf:
            outf.write(result_str + "\n")
        print(f"Ratio info written to: {args.output_txt}")

if __name__=="__main__":
    main()
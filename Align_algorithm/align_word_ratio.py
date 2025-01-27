import json
import csv

# 读取JSON文件
with open('Method4_small_WBW (3).json', 'r', encoding='utf-8') as f:
    word_by_word = json.load(f)

# 计算JSON中的总单词数
total_words = sum(len(segment['Words']) for segment in word_by_word if 'Words' in segment)

# 读取CSV文件并计算machine_text中的单词数
words_in_aligned = 0
with open('transcript_comparison_Method6.csv', 'r', encoding='utf-8') as f:
    csv_reader = csv.DictReader(f)
    for row in csv_reader:
        if row['machine_text'] and row['machine_text'].strip():
            # 分割文本并计算单词数
            words = row['machine_text'].split()
            words_in_aligned += len(words)

# 计算比率
ratio = (total_words - words_in_aligned) / total_words * 100
#ratio = (words_in_aligned / total_words) * 100

print(f"JSON文件中的总单词数: {total_words}")
print(f"CSV中机器转录文本的单词数: {words_in_aligned}")
print(f"比率: {ratio:.2f}%")
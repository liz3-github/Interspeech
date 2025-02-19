import pandas as pd
import re
import matplotlib.pyplot as plt
import os

def count_mixed_text(text):
    """
    Counts the total length of text, where 1 Chinese character = 1 unit and 1 English word = 1 unit.
    """
    if pd.isna(text):
        return 0
    chinese_chars = len(re.findall(r'[一-鿿]', text))
    english_words = len(re.findall(r'\b[a-zA-Z]+\b', text))
    return chinese_chars + english_words

def process_csv_files_with_counts(folder_path, output_chart, output_table):
    """
    Reads all CSV files in the folder, computes accuracy metrics with data amounts, 
    and generates a summary figure and table.
    """
    total_rows_with_text = 0
    total_rows_correct_text = 0
    total_rows = 0
    total_no_diff_rows = 0
    total_zero_timestamp_rows = 0
    total_text_difference = 0
    total_text_length = 0
    
    speaker_text_stats = {}
    language_text_stats = {}
    
    for filename in os.listdir(folder_path):
        if filename.endswith('.csv'):
            file_path = os.path.join(folder_path, filename)
            df = pd.read_csv(file_path)
            
            total_rows += len(df)
            total_no_diff_rows += df['Absolute_onset_time_difference'].isin(['00:00.00', '00:00:00', '00:00.0']).sum()
            total_zero_timestamp_rows += df['TimeStamp_difference'].isin(['00:00.00', '00:00:00', '00:00.0']).sum()
            total_text_difference += df['Text_difference_word_count'].sum()
            
            # Count text accuracy by entries
            valid_text_rows = df['Text_difference'].notna().sum()
            correct_text_rows = df['Text_difference'].eq(False).sum()
            total_rows_with_text += valid_text_rows
            total_rows_correct_text += correct_text_rows
            
            for _, row in df.iterrows():
                text_length = count_mixed_text(row['Text2'])  # Use human text as reference
                total_text_length += text_length
                
                # Process speakers
                human_speakers = set(str(row['Speaker2']).split(','))
                for speaker in human_speakers:
                    speaker = speaker.strip()
                    if speaker and speaker.lower() != 'nan':
                        if speaker not in speaker_text_stats:
                            speaker_text_stats[speaker] = {'total_text': 0, 'total_difference': 0}
                        speaker_text_stats[speaker]['total_text'] += text_length
                        speaker_text_stats[speaker]['total_difference'] += row['Text_difference_word_count']
                
                # Process languages
                languages = set(str(row['Language2']).split(','))
                for lang in languages:
                    lang = lang.strip()
                    if lang and lang.lower() != 'nan':
                        if lang not in language_text_stats:
                            language_text_stats[lang] = {'total_text': 0, 'total_difference': 0}
                        language_text_stats[lang]['total_text'] += text_length
                        language_text_stats[lang]['total_difference'] += row['Text_difference_word_count']
    
    # Compute overall accuracy
    onset_accuracy = total_no_diff_rows / total_rows if total_rows else 0
    timestamp_accuracy = total_zero_timestamp_rows / total_rows if total_rows else 0
    text_accuracy_by_entries = total_rows_correct_text / total_rows_with_text if total_rows_with_text else 0
    
    overall_speaker_accuracy = 1 - (sum(stats['total_difference'] for stats in speaker_text_stats.values()) / 
                                    sum(stats['total_text'] for stats in speaker_text_stats.values())) if speaker_text_stats else 0
    overall_language_accuracy = 1 - (sum(stats['total_difference'] for stats in language_text_stats.values()) / 
                                     sum(stats['total_text'] for stats in language_text_stats.values())) if language_text_stats else 0
    overall_text_accuracy = 1 - (total_text_difference / total_text_length) if total_text_length > 0 else 0
    
    accuracy_results = [
        ['Onset Time Accuracy', onset_accuracy, total_rows],
        ['Timestamp Accuracy', timestamp_accuracy, total_rows],
        ['Overall Speaker Accuracy', overall_speaker_accuracy, total_text_length],
        ['Overall Language Accuracy', overall_language_accuracy, total_text_length],
        ['Overall Text Accuracy', overall_text_accuracy, total_text_length],
        ['Text Accuracy by Entries', text_accuracy_by_entries, total_rows_with_text]
    ]
    
    # Add accuracy per speaker
    for speaker, stats in speaker_text_stats.items():
        accuracy_results.append([
            f'Speaker {speaker}', 
            1 - (stats['total_difference'] / stats['total_text']) if stats['total_text'] > 0 else 0, 
            stats['total_text']
        ])
    
    # Add accuracy per language
    for lang, stats in language_text_stats.items():
        accuracy_results.append([
            f'Language {lang}', 
            1 - (stats['total_difference'] / stats['total_text']) if stats['total_text'] > 0 else 0, 
            stats['total_text']
        ])
    
    # Convert to DataFrame and save
    result_df = pd.DataFrame(accuracy_results, columns=['Metric', 'Accuracy', 'Data Amount'])
    result_df.to_csv(output_table, index=False)
    
    # Create a bar chart
    plt.figure(figsize=(14, 7))
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2']
    bars = plt.bar(result_df['Metric'], result_df['Accuracy'], color=colors * (len(result_df) // len(colors) + 1))
    
    for bar, (acc, count) in zip(bars, zip(result_df['Accuracy'], result_df['Data Amount'])):
        plt.text(bar.get_x() + bar.get_width()/2, acc + 0.02, f'{acc:.2%}\n(n={count})', ha='center', fontsize=10)
    
    plt.ylabel('Accuracy')
    plt.title('Overall Accuracy Metrics with Data Amount')
    plt.xticks(rotation=45, ha='right')
    plt.ylim(0, 1.1)
    plt.tight_layout()
    plt.savefig(output_chart, dpi=300)
    plt.show()
    
    print("Bar chart saved as:", output_chart)
    print("Accuracy table saved as:", output_table)

if __name__ == "__main__":
    folder_path = 'processed_csv_folder'  # Replace with actual folder path
    output_chart = 'batch_accuracy_chart_with_data.png'
    output_table = 'batch_accuracy_table_with_data.csv'
    process_csv_files_with_counts(folder_path, output_chart, output_table)

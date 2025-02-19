import os
import pandas as pd

def anonymize_text_columns(folder_path, output_folder):
    """
    Reads all CSV files in the specified folder, removes the content in 'Text1' and 'Text2' columns,
    and saves the modified files into a new folder.
    """
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for filename in os.listdir(folder_path):
        if filename.endswith('.csv'):
            file_path = os.path.join(folder_path, filename)
            df = pd.read_csv(file_path)

            # Ensure 'Text1' and 'Text2' columns exist before modifying
            if 'Text1' in df.columns and 'Text2' in df.columns:
                df['Text1'] = ''
                df['Text2'] = ''

            output_file_path = os.path.join(output_folder, filename)
            df.to_csv(output_file_path, index=False)
            print(f"Anonymized file saved: {output_file_path}")

if __name__ == "__main__":
    folder_path = 'processed_csv_folder'  # Replace with the actual folder containing the files
    output_folder = 'anonymized_csv_folder'  # Folder to save modified files
    anonymize_text_columns(folder_path, output_folder)

import os
import pandas as pd
import Compute_diff_single as compute_diff

def process_all_csv_files(input_folder, output_folder):
    """Processes all CSV files in the input folder using Compute_diff_single.py logic."""
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    for filename in os.listdir(input_folder):
        if filename.endswith('.csv'):
            input_path = os.path.join(input_folder, filename)
            df = pd.read_csv(input_path)
            processed_df = compute_diff.calculate_differences(df)
            
            output_path = os.path.join(output_folder, filename)
            processed_df.to_csv(output_path, index=False)
            print(f"Processed and saved: {output_path}")

if __name__ == "__main__":
    input_folder = 'method6_azure'  # Folder containing raw CSV files
    output_folder = 'processed_csv_folder'  # Folder to store processed CSVs
    process_all_csv_files(input_folder, output_folder)
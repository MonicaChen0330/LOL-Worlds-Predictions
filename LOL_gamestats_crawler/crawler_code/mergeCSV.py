import os
import pandas as pd

# Define the input folder and output file
input_folder = input('Enter Input Folder Path:')
output_file = input("Enter Output Filename:")
output_path = output_file + '.csv'

# Initialize an empty list to store data from all CSV files
all_data = []

# Loop through all files in the folder
for filename in os.listdir(input_folder):
    if filename.endswith('.csv'):  # Only process CSV files
        file_path = os.path.join(input_folder, filename)
        print(f"Attempting to load: {file_path}")
        
        # Load the CSV file into a DataFrame
        try:
            df = pd.read_csv(file_path)
            if df.empty:
                print(f"Warning: {filename} is empty and will be skipped.")
                continue
            all_data.append(df)
            print(f"Loaded {filename} successfully. Shape: {df.shape}")
        except Exception as e:
            print(f"Failed to load {filename}: {e}")

# Combine all loaded DataFrames
if all_data:
    combined_df = pd.concat(all_data, ignore_index=True)
    
    # Save the combined DataFrame to the output CSV file
    try:
        combined_df.to_csv(output_path, index=False)
        print(f"All CSV files combined and saved to {output_path}.")
    except Exception as e:
        print(f"Failed to save combined CSV file: {e}")
else:
    print("No CSV files found or loaded.")

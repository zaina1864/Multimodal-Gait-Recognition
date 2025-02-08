import os
import pandas as pd
import numpy as np


# Base paths for saving data
raw_data_dir = r"new_data"
interpolated_data_dir = r"interpolated_data"
combined_data_dir = r"combined_data"


# Function to preprocess and interpolate data
def preprocess_and_interpolate(file_path, output_path):
    try:
        df = pd.read_csv(file_path)
        df.columns = [col.strip() for col in df.columns]
        if 'Timestamp (s)' in df.columns:
            df.rename(columns={'Timestamp (s)': 'Time'}, inplace=True)
        df['Time'] = pd.to_numeric(df['Time'], errors='coerce')
        df = df.dropna(subset=['Time']).reset_index(drop=True)
        first_time = df['Time'].iloc[0]
        df['Time'] = df['Time'] - first_time
        new_time = np.arange(df['Time'].min(), df['Time'].max(), 0.001)
        interpolated_df = pd.DataFrame({'Time': new_time})
        for col in df.columns:
            if col != 'Time':
                df[col] = pd.to_numeric(df[col], errors='coerce')
                interpolated_df[col] = np.interp(new_time, df['Time'], df[col])
        interpolated_df.to_csv(output_path, index=False)
        print(f"Processed and saved: {output_path}")
    except Exception as e:
        print(f"Failed to process {file_path}: {e}")

# Process all files for interpolation
def process_all_files(input_dir, output_dir, name):
    """
    Process only the folder that matches the given 'name'.
    - Checks if the subfolder matches 'name'
    - If yes, processes its CSV files
    """
    subfolder_path = os.path.join(input_dir, name)  # Check only the folder with the given name

    if os.path.isdir(subfolder_path):  # Ensure the folder exists
        output_subfolder_path = os.path.join(output_dir, name)  # Output folder with same name
        os.makedirs(output_subfolder_path, exist_ok=True)

        for file_name in os.listdir(subfolder_path):
            if file_name.endswith(".csv"):
                input_file_path = os.path.join(subfolder_path, file_name)
                output_file_path = os.path.join(output_subfolder_path, file_name)

                # Process and interpolate data
                preprocess_and_interpolate(input_file_path, output_file_path)

        print(f" Processed files for: {name}")
    else:
        print(f"Folder '{name}' not found in {input_dir}. No processing done.")



# Combine data
columns_order = [
    "Time", "ACCx_taR", "ACCy_taR", "ACCz_taR", "GYRx_taR", "GYRy_taR", "GYRz_taR",
    "ACCx_taL", "ACCy_taL", "ACCz_taL", "GYRx_taL", "GYRy_taL", "GYRz_taL"
]

column_mapping = {
    "Accel X (m/s^2)": "ACCx",
    "Accel Y (m/s^2)": "ACCy",
    "Accel Z (m/s^2)": "ACCz",
    "Gyro X (rad/s)": "GYRx",
    "Gyro Y (rad/s)": "GYRy",
    "Gyro Z (rad/s)": "GYRz",
}

def preprocess_file(file_path, sensor_suffix):
    try:
        df = pd.read_csv(file_path)
        df.rename(columns=column_mapping, inplace=True)
        if sensor_suffix:
            df = df.rename(columns={col: f"{col}_{sensor_suffix}" for col in df.columns if col != "Time"})
        return df
    except Exception as e:
        print(f"Failed to preprocess {file_path}: {e}")
        return None

def remove_empty_rows(df):
    non_time_columns = df.columns.difference(["Time"])
    df = df.dropna(how="any", subset=non_time_columns)
    return df


def combine_data(subject_folder):
    try:
        taL_path, taR_path = None, None

        for file in os.listdir(subject_folder):
            if file.startswith("taL_") and file.endswith(".csv"):
                taL_path = os.path.join(subject_folder, file)
            elif file.startswith("taR_") and file.endswith(".csv"):
                taR_path = os.path.join(subject_folder, file)

        taL = preprocess_file(taL_path, "taL")
        taR = preprocess_file(taR_path, "taR")

        if any(df is None for df in [taL, taR]):
            print(f"Skipping {subject_folder} due to preprocessing errors.")
            return
        combined_df = taR
        combined_df = combined_df.merge(taL, on="Time", how="outer")
        combined_df = remove_empty_rows(combined_df)
        combined_df = combined_df[columns_order]
        subject_name = os.path.basename(subject_folder)
        output_file = os.path.join(combined_data_dir, f"{subject_name}.csv")
        combined_df.to_csv(output_file, index=False)
        print(f"Combined data saved to: {output_file}")
    except Exception as e:
        print(f"Failed to process {subject_folder}: {e}")

def process_all_subjects(interpolated_dir,name):
    subject_path = os.path.join(interpolated_dir, name)
    if os.path.isdir(subject_path):
        combine_data(subject_path)
        print(f"Processed data for subject: {name}")
    else:
        print(f"Error: Subject folder '{name}' not found.")


def process_sensor_data(name):
    process_all_files(raw_data_dir, interpolated_data_dir,name)
    process_all_subjects(interpolated_data_dir,name)


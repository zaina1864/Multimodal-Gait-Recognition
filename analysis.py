import numpy as np
import pandas as pd

from scipy.signal import find_peaks, savgol_filter
import matplotlib.pyplot as plt
from gait_analysis_video import GaitAnalysis


# Function to refine troughs
def refine_troughs(signal, troughs, window_size=3):
    refined_troughs = []
    for trough in troughs:
        start = max(0, trough - window_size)
        end = min(len(signal), trough + window_size + 1)
        local_min = np.argmax(signal[start:end])
        refined_trough = start + local_min
        if refined_trough not in refined_troughs:
            refined_troughs.append(refined_trough)
    return refined_troughs


# Function to extract events
def extract_events(time, signal, peaks, troughs, heel_strike_label, toe_off_label):
    troughs = np.array(troughs)
    heel_strikes = []
    toe_offs = []

    for peak in peaks:
        troughs_before = troughs[troughs < peak]
        troughs_after = troughs[troughs > peak]

        if len(troughs_before) > 0 and len(troughs_after) > 0:
            toe_off = troughs_before[-1]
            heel_strike = troughs_after[0]

            if signal[toe_off] < 0 and signal[heel_strike] < 0:
                toe_offs.append(toe_off)
                heel_strikes.append(heel_strike)

    marked_troughs = set(heel_strikes + toe_offs)
    unmarked_troughs = [t for t in troughs if t not in marked_troughs]

    for trough in unmarked_troughs:
        if trough < (toe_offs[0] if toe_offs else len(signal)):
            heel_strikes.insert(0, trough)
        elif trough > (heel_strikes[-1] if heel_strikes else 0):
            toe_offs.append(trough)

    return {
        heel_strike_label: pd.Series(heel_strikes).reset_index(drop=True),
        toe_off_label: pd.Series(toe_offs).reset_index(drop=True),
    }


def calculate_gait_features(combined_events, original_data):
    """
    Compute gait features including stance, swing, step times,


    Parameters:
    - combined_events: DataFrame with detected gait events.
    - original_data: DataFrame with raw sensor timestamps.

    Returns:
    - features: DataFrame containing computed gait features.
    """

    # Extract indices of detected gait events
    RHS_indices = combined_events["RHS"].dropna().astype(int).values
    RTO_indices = combined_events["RTO"].dropna().astype(int).values
    LHS_indices = combined_events["LHS"].dropna().astype(int).values
    LTO_indices = combined_events["LTO"].dropna().astype(int).values

    # Retrieve corresponding timestamps
    RHS_times = original_data.iloc[RHS_indices]["Time"].values
    RTO_times = original_data.iloc[RTO_indices]["Time"].values
    LHS_times = original_data.iloc[LHS_indices]["Time"].values
    LTO_times = original_data.iloc[LTO_indices]["Time"].values

    # Compute Stride Time (Time between two consecutive heel strikes of the same foot)
    stride_time_left = np.abs(np.diff(LHS_times))  # Time difference between left heel strikes
    stride_time_right = np.abs(np.diff(RHS_times))  # Time difference between right heel strikes

    # Ensure step durations are computed with matching array lengths
    min_len_step = min(len(LHS_times), len(RHS_times))

    step_durations_left = np.abs(LHS_times[:min_len_step] - RHS_times[:min_len_step])
    step_durations_right = np.abs(RHS_times[:min_len_step] - LHS_times[:min_len_step])

    # Compute Swing Time (Time between toe-off and next heel strike)
    swing_time_left = np.abs(LHS_times[1:] - LTO_times[:len(LHS_times) - 1]) if len(LHS_times) > 1 else []
    swing_time_right = np.abs(RHS_times[1:] - RTO_times[:len(RHS_times) - 1]) if len(RHS_times) > 1 else []

    # Compute Stance Time (Time between heel strike and the following toe-off)
    min_len_stance_left = min(len(LHS_times), len(LTO_times))
    min_len_stance_right = min(len(RHS_times), len(RTO_times))

    stance_time_left = np.abs(LTO_times[:min_len_stance_left] - LHS_times[:min_len_stance_left])
    stance_time_right = np.abs(RTO_times[:min_len_stance_right] - RHS_times[:min_len_stance_right])

    # Compute Single Support Time (Step Duration - Swing Time)
    min_len_support_left = min(len(step_durations_left), len(swing_time_left))
    min_len_support_right = min(len(step_durations_right), len(swing_time_right))

    single_support_left = np.abs(step_durations_left[:min_len_support_left] - swing_time_left[:min_len_support_left])
    single_support_right = np.abs(
        step_durations_right[:min_len_support_right] - swing_time_right[:min_len_support_right])

    # Ensure all arrays have the same length before creating the DataFrame
    max_len = max(len(stride_time_left), len(stride_time_right),
                  len(swing_time_left), len(swing_time_right),
                  len(stance_time_left), len(stance_time_right),
                  len(step_durations_left), len(step_durations_right),
                  len(single_support_left), len(single_support_right))

    def pad_and_round(arr, max_length):
        return np.round(np.pad(arr, (0, max_length - len(arr)), constant_values=np.nan), 4)

    stride_time_left = pad_and_round(stride_time_left, max_len)
    stride_time_right = pad_and_round(stride_time_right, max_len)
    single_support_left = pad_and_round(single_support_left, max_len)
    single_support_right = pad_and_round(single_support_right, max_len)

    # Calculate cadence (steps per minute)
    avg_step_time = np.nanmean([np.nanmean(stride_time_left), np.nanmean(stride_time_right)])
    cadence = round((60 / avg_step_time) if avg_step_time > 0 else np.nan, 4)

    # Create DataFrame for computed gait features
    features = pd.DataFrame({
        "Stride Time (L)": stride_time_left,
        "Stride Time (R)": stride_time_right,
        "Single Support (L)": single_support_left,
        "Single Support (R)": single_support_right,
        "Cadence (steps/min)": [cadence] * max_len,
    })

    return features


# Function to plot the signal and events
def plot_signal(time, original_signal, smoothed_signal, peaks, heel_strikes, toe_offs):
    plt.figure(figsize=(14, 8))
    plt.plot(time, original_signal, label="Original Signal", alpha=0.6)
    plt.plot(time, smoothed_signal, label="Smoothed Signal", linewidth=2)
    plt.scatter(time.iloc[peaks], smoothed_signal[peaks], color="red", label="Midswing (Peaks)", zorder=5)
    plt.scatter(time.iloc[heel_strikes], smoothed_signal[heel_strikes], color="blue", label="Heel Strike", zorder=5)
    plt.scatter(time.iloc[toe_offs], smoothed_signal[toe_offs], color="green", label="Toe Off", zorder=5)
    plt.xlabel("Time (ms)")
    plt.ylabel("Gyro Y (rad/s)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    return plt

def process_file(file):
    data = pd.read_csv(file)
    time = pd.to_numeric(data["Time"], errors="coerce") * 1000
    gyro_y_right = data["GYRy_taR"]
    gyro_y_left = data["GYRy_taL"]

    gyro_y_right_smoothed = savgol_filter(gyro_y_right, window_length=51, polyorder=3)
    gyro_y_left_smoothed = savgol_filter(gyro_y_left, window_length=51, polyorder=3)

    peaks_right, _ = find_peaks(gyro_y_right_smoothed, height=1, distance=500)
    troughs_right, _ = find_peaks(-gyro_y_right_smoothed, height=0.3, distance=45)
    troughs_right_refined = refine_troughs(gyro_y_right_smoothed, troughs_right, window_size=5)

    peaks_left, _ = find_peaks(gyro_y_left_smoothed, height=1, distance=500)
    troughs_left, _ = find_peaks(-gyro_y_left_smoothed, height=0.3, distance=45)
    troughs_left_refined = refine_troughs(gyro_y_left_smoothed, troughs_left, window_size=5)

    gait_events_right = extract_events(
        time, gyro_y_right_smoothed, peaks_right, troughs_right_refined, "RHS", "RTO"
    )
    gait_events_left = extract_events(
        time, gyro_y_left_smoothed, peaks_left, troughs_left_refined, "LHS", "LTO"
    )

    # Count steps for both feet
    steps_right = len(gait_events_right["RHS"].dropna())
    steps_left = len(gait_events_left["LHS"].dropna())
    total_steps = steps_right + steps_left

    combined_events = pd.DataFrame(
        {
            "RHS": gait_events_right["RHS"].reindex(range(max(len(gait_events_right["RHS"]), len(gait_events_left["LHS"])))),
            "RTO": gait_events_right["RTO"].reindex(range(max(len(gait_events_right["RHS"]), len(gait_events_left["LHS"])))),
            "LHS": gait_events_left["LHS"].reindex(range(max(len(gait_events_right["RHS"]), len(gait_events_left["LHS"])))),
            "LTO": gait_events_left["LTO"].reindex(range(max(len(gait_events_right["RHS"]), len(gait_events_left["LHS"])))),
        }
    ).reset_index(drop=True)

    gait_features = calculate_gait_features(combined_events, data)

    plt_right = plot_signal(
        time, gyro_y_right, gyro_y_right_smoothed, peaks_right, gait_events_right["RHS"], gait_events_right["RTO"]
    )

    plt_left = plot_signal(
        time, gyro_y_left, gyro_y_left_smoothed, peaks_left, gait_events_left["LHS"], gait_events_left["LTO"]
    )

    return plt_right, plt_left, combined_events, gait_features, total_steps


def process_gait(file):
    plt_right, plt_left, combined_events, gait_features, total_steps = process_file(file)
    plt_right.savefig("right_plot.png")
    plt_left.savefig("left_plot.png")
    combined_events.to_csv("gait_events.csv", index=False)

    return "right_plot.png", "left_plot.png", gait_features, total_steps


def process_video(file):
    """
    Process the uploaded video file for gait analysis and count steps.
    """
    # Path to MediaPipe pose model
    model_path = "model\\pose_landmarker_heavy.task"
    video_path = file

    # Initialize GaitAnalysis and process the video
    gait_analyzer = GaitAnalysis(video_path, model_path)
    output_video_path, df, result, plot1, plot2 = gait_analyzer.process_video()
    df = df.round(4)
    # Save the DataFrame and plots
    df.to_csv("video_gait_data.csv", index=False)
    plot1.savefig("right.png")
    plot2.savefig("left.png")

    # Calculate total steps from the video data (count rows in the DataFrame, assuming 1 row per step)
    total_steps = len(df) * 2
    # Create Stance and Swing Time Plots
    stance_plot_path = "stance_plot.png"

    plt.figure(figsize=(12, 6))
    # Stance Times
    plt.subplot(1, 2, 1)
    plt.plot(df["Stance Time (L)"], label="Stance Time Left", marker="o", color="blue")
    plt.plot(df["Stance Time (R)"], label="Stance Time Right", marker="o", color="red")
    plt.title("Stance Times (Left vs Right)")
    plt.xlabel("Steps")
    plt.ylabel("Time (s)")
    plt.legend()
    plt.grid(True)

    # Swing Times
    plt.subplot(1, 2, 2)
    plt.plot(df["Swing Time (L)"], label="Swing Time Left", marker="o", color="blue")
    plt.plot(df["Swing Time (R)"], label="Swing Time Right", marker="o", color="red")
    plt.title("Swing Times (Left vs Right)")
    plt.xlabel("Steps")
    plt.ylabel("Time (s)")
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig(stance_plot_path)
    plt.close()

    return total_steps, output_video_path, df, "right.png", "left.png", stance_plot_path


import mediapipe as mp
import cv2
import numpy as np
import pandas as pd
from scipy.signal import find_peaks, butter, filtfilt
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
from mediapipe.framework.formats import landmark_pb2
from mediapipe import solutions
import os
import uuid


class GaitAnalyzer:
    def __init__(self, video_source, model_file="./model/pose_landmarker_heavy.task"):
        self.video_source = video_source
        self.model_file = model_file
        self.landmarker_config = self.setup_landmarker()
        self.frame_rate = None

    def setup_landmarker(self):
        options = mp.tasks.vision.PoseLandmarkerOptions(
            base_options=mp.tasks.BaseOptions(model_asset_path=self.model_file),
            running_mode=mp.tasks.vision.RunningMode.VIDEO,
            min_pose_detection_confidence=0.5
        )
        return mp.tasks.vision.PoseLandmarker.create_from_options(options)

    @staticmethod
    def render_landmarks(image_rgb, detection_output):
        landmark_groups = detection_output.pose_landmarks
        processed_image = np.copy(image_rgb)

        for index in range(len(landmark_groups)):
            body_landmarks = landmark_groups[index]
            pose_proto = landmark_pb2.NormalizedLandmarkList()
            pose_proto.landmark.extend([
                landmark_pb2.NormalizedLandmark(x=point.x, y=point.y, z=point.z) for point in body_landmarks
            ])
            solutions.drawing_utils.draw_landmarks(
                processed_image,
                pose_proto,
                solutions.pose.POSE_CONNECTIONS,
                solutions.drawing_styles.get_default_pose_landmarks_style()
            )
        return processed_image

    @staticmethod
    def export_video(annotated_frames, fps):
        output_dir = "output_videos"
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        video_filename = uuid.uuid4().hex
        output_video_path = os.path.join(output_dir, video_filename + ".webm")

        if len(annotated_frames[0].shape) == 2:
            height, width = annotated_frames[0].shape
            annotated_frames = [cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR) for frame in annotated_frames]
        else:
            height, width, _ = annotated_frames[0].shape

        codec = cv2.VideoWriter_fourcc(*'vp80')
        video_writer = cv2.VideoWriter(output_video_path, codec, fps, (width, height))

        for frame in annotated_frames:
            video_writer.write(frame)
        video_writer.release()

        return output_video_path

    @staticmethod
    def interpolate_gaps(left_foot, right_foot):
        x_values = np.arange(len(left_foot))
        left_interp = interp1d(x_values, left_foot, kind='cubic', fill_value="extrapolate")
        right_interp = interp1d(x_values, right_foot, kind='cubic', fill_value="extrapolate")
        return left_interp(x_values), right_interp(x_values)

    @staticmethod
    def apply_low_pass_filter(left_data, right_data, fps):
        sampling_freq = len(left_data) / fps
        nyquist_freq = 0.5 * sampling_freq
        cutoff_freq = 0.875
        filter_order = 10
        normalized_cutoff = cutoff_freq / nyquist_freq
        b_coeff, a_coeff = butter(filter_order, normalized_cutoff, btype='low', analog=False)
        return filtfilt(b_coeff, a_coeff, left_data), filtfilt(b_coeff, a_coeff, right_data)
    # Process video and calculate gait
    def process_video(self):
        with self.pose_landmarker_options as landmarker:
            cap = cv2.VideoCapture(self.video_path)
            frame_rate = int(cap.get(cv2.CAP_PROP_FPS))
            frame_number = 0
            annotated_frames = []
            left_distance, right_distance = [], []

            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break  # Exit the loop if no more frames are available

                # Convert the frame received from OpenCV to a MediaPipe’s Image object.
                numpy_frame_from_opencv = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=numpy_frame_from_opencv)

                # Calculate the timestamp for the current frame
                frame_timestamp_ms = int(frame_number * (1000 / frame_rate))

                # Perform pose landmarking on the provided single image.
                pose_landmarker_result = landmarker.detect_for_video(mp_image, frame_timestamp_ms)
                annotated_image = self.render_landmarks(frame, pose_landmarker_result)
                annotated_frames.append(annotated_image)

                if pose_landmarker_result.pose_landmarks:
                    landmarks = pose_landmarker_result.pose_landmarks[0]
                    body_coords = [(landmark.x, landmark.y, landmark.z) for landmark in landmarks]

                    # Get keypoints and their visibility
                    left_hip = np.array(body_coords[23])
                    right_hip = np.array(body_coords[24])
                    left_foot_index = np.array(body_coords[31])
                    right_foot_index = np.array(body_coords[32])

                    left_distance.append(np.linalg.norm(np.subtract(left_hip, left_foot_index)))
                    right_distance.append(np.linalg.norm(np.subtract(right_hip, right_foot_index)))

                frame_number += 1

            cap.release()

            left_distance_filled, right_distance_filled = self.interpolate_gaps(left_distance, right_distance)

            left_distance_filtered, right_distance_filtered = self.apply_low_pass_filter(left_distance_filled,
                                                                                       right_distance_filled, frame_rate)

            # Find peaks for heel strike
            peaks_left, _ = find_peaks(left_distance_filtered, distance=0.8 * frame_rate)
            peaks_right, _ = find_peaks(right_distance_filtered, distance=0.8 * frame_rate)

            # Find minimum for toe-off
            minimum_left, _ = find_peaks(-left_distance_filtered, distance=0.8 * frame_rate)
            minimum_right, _ = find_peaks(-right_distance_filtered, distance=0.8 * frame_rate)

            # Plotting distances, peaks and minimum
            # For Left Leg
            plt1 = plt.figure(1, figsize=(15, 6))
            plt.plot(left_distance_filtered, label=" Left Leg", color="blue")
            plt.scatter(peaks_left, [left_distance_filtered[i] for i in peaks_left], color="red",
                        label="Heel Strikes Left Leg")
            plt.scatter(minimum_left, [left_distance_filtered[i] for i in minimum_left], color="green",
                        label="Toe-offs Left Leg")
            plt.title("Distances, Peaks (Heel Strikes), and minimum (Toe-offs) for Left Leg")
            plt.xlabel("Frame Number")
            plt.ylabel("Distance")
            plt.legend()
            plt.grid(True)

            # For Right Leg
            plt2 = plt.figure(2, figsize=(15, 6))
            plt.plot(right_distance_filtered, label=" Right Leg", color="blue")
            plt.scatter(peaks_right, [right_distance_filtered[i] for i in peaks_right], color="red",
                        label="Heel Strikes Right Leg")
            plt.scatter(minimum_right, [right_distance_filtered[i] for i in minimum_right], color="green",
                        label="Toe-offs Right Leg")
            plt.title("Distances, Peaks (Heel Strikes), and minimum (Toe-offs) for Right Leg")
            plt.xlabel("Frame Number")
            plt.ylabel("Distance")
            plt.legend()
            plt.grid(True)

            # Calculate stance times for the right leg
            stance_times_right = []
            for i in range(len(peaks_right)):
                # Find the subsequent toe-off after the current heel strike
                subsequent_minimum = [minimum for minimum in minimum_right if minimum > peaks_right[i]]

                # If there is a subsequent toe-off, calculate stance time
                if subsequent_minimum:
                    stance_time = (subsequent_minimum[0] - peaks_right[i]) / frame_rate
                    stance_time = abs(stance_time)
                    stance_times_right.append(stance_time)

            # Calculate stance times for the left leg
            stance_times_left = []
            for i in range(len(peaks_left)):
                # Find the subsequent toe-off after the current heel strike
                subsequent_minimum = [minimum for minimum in minimum_left if minimum > peaks_left[i]]

                # If there is a subsequent toe-off, calculate stance time
                if subsequent_minimum:
                    stance_time = (subsequent_minimum[0] - peaks_left[i]) / frame_rate
                    stance_time = abs(stance_time)
                    stance_times_left.append(stance_time)
            # Swing Time for left foot
            try:
                swing_time_left = [(peaks_left[i + 1] - minimum_left[i]) / frame_rate for i in
                                   range(len(minimum_left) - 1)]
            except IndexError:
                swing_time_left = [(peaks_left[i + 1] - minimum_left[i]) / frame_rate for i in
                                   range(min(len(peaks_left) - 1, len(minimum_left)))]


            # Swing Time for right foot
            try:
                swing_time_right = [(peaks_right[i + 1] - minimum_right[i]) / frame_rate for i in
                                    range(len(minimum_right) - 1)]

            except IndexError:
                swing_time_right = [(peaks_right[i + 1] - minimum_right[i]) / frame_rate for i in
                                    range(min(len(peaks_right) - 1, len(minimum_right)))]

            # Step Time for left foot
            try:
                step_time_left = [(peaks_left[i + 1] - peaks_left[i]) / frame_rate for i in range(len(peaks_left) - 1)]
            except IndexError:
                step_time_left = [(peaks_left[i + 1] - peaks_left[i]) / frame_rate for i in range(len(peaks_left) - 2)]

            # Step Time for right foot
            try:
                step_time_right = [(peaks_right[i + 1] - peaks_right[i]) / frame_rate for i in
                                   range(len(peaks_right) - 1)]
            except IndexError:
                step_time_right = [(peaks_right[i + 1] - peaks_right[i]) / frame_rate for i in
                                   range(len(peaks_right) - 2)]

            max_len = max(len(stance_times_left), len(stance_times_right),
                          len(swing_time_left), len(swing_time_right),
                          len(step_time_left), len(step_time_right),
                          )
            
            def pad_list(lst, max_len, pad_value=np.nan):
                return lst + [pad_value] * (max_len - len(lst))

            # Save results to a dataframe
            self.df = pd.DataFrame({
                'Stance Time (L)': pad_list(stance_times_left, max_len),
                'Stance Time (R)': pad_list(stance_times_right, max_len),
                'Swing Time (L)': pad_list(swing_time_left, max_len),
                'Swing Time (R)': pad_list(swing_time_right, max_len),
                'Stride Time (L)': pad_list(step_time_left, max_len),
                'Stride Time (R)': pad_list(step_time_right, max_len),
            })

            # Store the results in string
            result = "Stance Time Left: {stance_time_left}, Stance Time Right: {stance_time_right}, Swing Time Left: {swing_time_left}, Swing Time Right: {swing_time_right}, Step Time Left: {step_time_left}, Step Time Right: {step_time_right}".format(
                stance_time_left=stance_times_left,
                swing_time_left=swing_time_left,
                stance_time_right=stance_times_right,
                swing_time_right=swing_time_right,
                step_time_left=step_time_left,
                step_time_right=step_time_right,
            )

        output_video_path = self.save_annotated_video(annotated_frames, frame_rate)

        return output_video_path, self.df, result, plt1, plt2




import cv2
import time
import numpy as np
import pandas as pd
import math
from tqdm import tqdm
from tensorflow.keras.models import load_model
from tensorflow.keras import backend as K
import csv
import ast


def flatten_list(nested_list):
    flat_list = []
    for item in nested_list:
        if item is None:
            flat_list.extend([0, 0])
        else:
            try:
                flat_list.extend(item)
            except:
                flat_list.append(item)
    return flat_list

def calculate_distance(x1, y1, x2, y2):
    return math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)

def normalize_joint_positions(joint_positions):
    normalized_positions = [0] * 36
    reference_x, reference_y, reference_length = 0, 0, 1

    if joint_positions[2] != 0 or joint_positions[3] != 0:
        reference_x, reference_y = joint_positions[2], joint_positions[3]
    elif (joint_positions[4] != 0 or joint_positions[5] != 0) and (joint_positions[10] != 0 or joint_positions[11] != 0):
        reference_x = (joint_positions[4] + joint_positions[10]) / 2
        reference_y = (joint_positions[5] + joint_positions[11]) / 2
    elif joint_positions[4] != 0 or joint_positions[5] != 0:
        reference_x, reference_y = joint_positions[4], joint_positions[5]
    elif joint_positions[10] != 0 or joint_positions[11] != 0:
        reference_x, reference_y = joint_positions[10], joint_positions[11]

    if joint_positions[16] != 0 and joint_positions[17] != 0:
        reference_length = calculate_distance(joint_positions[16], joint_positions[17], reference_x, reference_y)
    elif joint_positions[22] != 0 and joint_positions[23] != 0:
        reference_length = calculate_distance(joint_positions[22], joint_positions[23], reference_x, reference_y)

    for i in range(18):
        x_index, y_index = i * 2, i * 2 + 1
        if joint_positions[x_index] != 0:
            normalized_positions[x_index] = (joint_positions[x_index] - reference_x) / reference_length
        else:
            normalized_positions[x_index] = joint_positions[x_index]

        if joint_positions[y_index] != 0:
            normalized_positions[y_index] = -(joint_positions[y_index] - reference_y) / reference_length
        else:
            normalized_positions[y_index] = joint_positions[y_index]

    return normalized_positions

def adjust_detected_frames(max_frames, pose_data):
    detected_frames = pose_data.shape[0]
    if detected_frames > max_frames:
        return pose_data[:max_frames, :]
    else:
        zero_padded_array = np.zeros((max_frames, 36))
        zero_padded_array[:pose_data.shape[0], :pose_data.shape[1]] = pose_data
        return zero_padded_array

def extract_body_pose(video_file, neural_net):
    body_pose_data = []
    video_capture = cv2.VideoCapture(video_file)
    max_no_keypoints_frames = 10
    frames_without_keypoints = max_no_keypoints_frames

    while True:
        success, frame = video_capture.read()
        if not success:
            break

        frame_width, frame_height = frame.shape[1], frame.shape[0]
        blob = cv2.dnn.blobFromImage(frame, 1.0 / 255, (368, 368), (0, 0, 0), swapRB=False, crop=False)
        neural_net.setInput(blob)
        output = neural_net.forward()

        height, width = output.shape[2], output.shape[3]
        keypoints = []

        for i in range(18):
            probability_map = output[0, i, :, :]
            _, prob, _, coord = cv2.minMaxLoc(probability_map)
            x_coord = int((frame_width * coord[0]) / width)
            y_coord = int((frame_height * coord[1]) / height)

            if prob > 0.1:
                keypoints.append((x_coord, y_coord))
            else:
                keypoints.append(None)

        processed_keypoints = flatten_list(keypoints)
        body_pose_data.append(processed_keypoints)

        if all(point is None for point in keypoints) and frames_without_keypoints < 1:
            break
        if len(body_pose_data) > 95:
            break

    video_capture.release()
    return body_pose_data


def load_x(data):
    return np.array([[ast.literal_eval(i) if isinstance(i, str) else i for i in row] for row in tqdm(data.itertuples(index=False))])


def load_y(data):
    return np.array([ast.literal_eval(row[0]) if isinstance(row[0], str) else row[0] for row in tqdm(data.itertuples(index=False))])


def assign_new_person_label(df_label):
    y_labels = []
    one_hot_encoded_labels = []
    new_label = len(ast.literal_eval(df_label.loc[0].tolist()[0])) + 1

    for row in tqdm(df_label.itertuples(index=False)):
        person_id = (ast.literal_eval(row[0])).index(max(ast.literal_eval(row[0]))) + 1
        y_labels.append(person_id)

    y_labels.append(new_label)
    total_classes = len(set(y_labels))

    for value in y_labels:
        one_hot_vector = [0] * total_classes
        one_hot_vector[int(value) - 1] = 1  # Adjusting for zero-based index
        one_hot_encoded_labels.append(one_hot_vector)

    return np.array(one_hot_encoded_labels), new_label


def onboard_user(video_file, name, age, gender):
    global df, df_label
    from tensorflow.keras import backend as K
    proto_file = "model/pose_deploy_linevec.prototxt"
    weights_file = "model/pose_iter_440000.caffemodel"
    data_augmented_path = "static/data_augmented.csv"
    label_augmented_path = "static/PersonLabel_augmented.csv"

    df = pd.read_csv(data_augmented_path, header=None)
    df_label = pd.read_csv(label_augmented_path, header=None)

    net = cv2.dnn.readNetFromCaffe(proto_file, weights_file)

    try:
        pose_data = extract_body_pose(video_file, net)
        normalized_pose_data = [normalize_joint_positions(frame) for frame in pose_data]
        pose_array = np.array(normalized_pose_data)

        max_frames = 95
        processed_data = adjust_detected_frames(max_frames, pose_array)
        processed_df = pd.DataFrame(processed_data)
        processed_df.columns = [str(i) for i in range(df.shape[1])]
        df.columns = processed_df.columns

        df = pd.concat([df, processed_df], axis=0, ignore_index=True)

        y_onehot, new_id = assign_new_person_label(df_label)
        df_label = y_onehot

        x_data = load_x(df)
        y_data = df_label

        x_data_path = 'static/data_augmented.csv'
        with open(x_data_path, 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerows(x_data.tolist())

        y_df = pd.DataFrame([list(row) for row in y_data])
        y_df = y_df.applymap(str)
        y_df = y_df.apply(lambda row: '[' + ', '.join(row) + ']', axis=1).to_frame()

        y_data_path = 'static/PersonLabel_augmented.csv'
        y_df.to_csv(y_data_path, index=False, header=False)

        mapping_file = "static/ID_Name.csv"
        with open(mapping_file, 'a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([new_id, name, age, gender])

        return f"Onboarding complete. Data saved to {x_data_path} and {y_data_path}."

    except Exception as e:
        return f"An error occurred during onboarding: {e}"


def onboard_user(video_file, name, age, gender):
    global df, df_label
    from tensorflow.keras import backend as K
    proto_file = "model/pose_deploy_linevec.prototxt"
    weights_file = "model/pose_iter_440000.caffemodel"
    data_augmented_path = "static/data_augmented.csv"
    label_augmented_path = "static/PersonLabel_augmented.csv"

    df = pd.read_csv(data_augmented_path, header=None)
    df_label = pd.read_csv(label_augmented_path, header=None)

    net = cv2.dnn.readNetFromCaffe(proto_file, weights_file)

    try:
        pose_data = extract_body_pose(video_file, net)
        normalized_pose_data = [normalize_joint_positions(frame) for frame in pose_data]
        pose_array = np.array(normalized_pose_data)

        max_frames = 95
        processed_data = adjust_detected_frames(max_frames, pose_array)
        processed_df = pd.DataFrame(processed_data)
        processed_df.columns = [str(i) for i in range(df.shape[1])]
        df.columns = processed_df.columns

        df = pd.concat([df, processed_df], axis=0, ignore_index=True)

        y_onehot, new_id = assign_new_person_label(df_label)
        df_label = y_onehot

        x_data = load_x(df)
        y_data = df_label

        x_data_path = 'static/data_augmented.csv'
        with open(x_data_path, 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerows(x_data.tolist())

        y_df = pd.DataFrame([list(row) for row in y_data])
        y_df = y_df.applymap(str)
        y_df = y_df.apply(lambda row: '[' + ', '.join(row) + ']', axis=1).to_frame()

        y_data_path = 'static/PersonLabel_augmented.csv'
        y_df.to_csv(y_data_path, index=False, header=False)

        mapping_file = "static/ID_Name.csv"
        with open(mapping_file, 'a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([new_id, name, age, gender])

        return f"Onboarding complete. Data saved to {x_data_path} and {y_data_path}."

    except Exception as e:
        return f"An error occurred during onboarding: {e}"


def generate_prediction(model_path, detected_data, reference_data, labels):
    max_frames = 95
    model = load_model(model_path, compile=False)
    label_classes = [i + 1 for i in np.argmax(labels, axis=1)]
    similarity_scores = []
    prediction_results = []

    detected_data = detected_data.reshape((1, max_frames, 36))

    for index in tqdm(range(len(reference_data))):
        sample = reference_data[index][:max_frames, :36].reshape((1, max_frames, 36))
        similarity = model.predict([detected_data, sample])[0][0]
        similarity_scores.append(similarity)
        prediction_results.append((label_classes[index], similarity))

    best_match_index = similarity_scores.index(max(similarity_scores))
    best_score = similarity_scores[best_match_index]
    predicted_person = label_classes[best_match_index]

    if np.isnan(best_score):
        identity = 'Not Detected'
        message = 'No Person Detected.'
    elif best_score >= 0.72:
        identity = predicted_person
        message = 'This Person Is Authorized To Enter'
    else:
        identity = 'A Person Detected'
        message = 'NOT AUTHORIZED. DO NOT ENTER!'

    print(f'Predicted ID: {identity}\t Message: {message}\t Similarity Score: {best_score}')
    return predicted_person, message, prediction_results


def verify_person(test_video):
    global df, df_label, processed_data
    proto_file = "model/pose_deploy_linevec.prototxt"
    weights_file = "model/pose_iter_440000.caffemodel"
    data_augmented_path = "static/data_augmented.csv"
    label_augmented_path = "static/PersonLabel_augmented.csv"
    model_file_path = "model/siamesenetwork.h5"
    mapping_file = "static/ID_Name.csv"

    net = cv2.dnn.readNetFromCaffe(proto_file, weights_file)

    df = pd.read_csv(data_augmented_path, header=None)
    df_label = pd.read_csv(label_augmented_path, header=None)

    x = load_x(df)
    y = load_y(df_label) if not isinstance(df_label, np.ndarray) else df_label

    body_pose_list = extract_body_pose(test_video, net)
    normalized_body_pose_list = [normalize_joint_positions(frame) for frame in body_pose_list]
    body_pose_array = np.array(normalized_body_pose_list)

    max_frames = 95
    processed_data = adjust_detected_frames(max_frames, body_pose_array)

    x_sample = x[-20:, :max_frames, :36]
    y_sample = y[-20:]

    predicted_class, message, scores_and_ids = generate_prediction(model_file_path, processed_data, x_sample, y_sample)

    name, age, gender = "Unknown", "Unknown", "Unknown"

    try:
        mapping_df = pd.read_csv(mapping_file)
        mapping_df["ID"] = mapping_df["ID"].astype(str)
        name_row = mapping_df[mapping_df["ID"] == str(predicted_class)]

        if not name_row.empty:
            name = name_row["Name"].values[0]
            age = name_row["Age"].values[0]
            gender = name_row["Gender"].values[0]

    except Exception as e:
        print(f"Error retrieving details: {e}")

    scores_table = pd.DataFrame(scores_and_ids, columns=["ID", "Similarity Score"])
    scores_table.sort_values(by="Similarity Score", ascending=False, inplace=True)

    output_message = (
        f"Verified Person ID: {predicted_class}\n"
        f"Message: {message}\n"
        f"Name: {name}\n"
        f"Age: {age}\n"
        f"Gender: {gender}"
    )
    return output_message, scores_table





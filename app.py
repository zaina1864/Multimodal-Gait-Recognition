import subprocess
from flask import Flask, url_for, render_template, request, redirect, session, Response, jsonify
from flask_sqlalchemy import SQLAlchemy 
import serial
import os
import threading
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from collections import defaultdict

from sensor_processing import process_sensor_data
from onboard import onboard_user, verify_person
from tensorflow.keras.models import load_model
from tensorflow.keras import backend as K
import cv2
import time
from analysis import process_video, process_gait
import csv


# Configuration for Bluetooth COM ports and baud rate
right_imu_port = 'COM6'  # Replace with your Bluetooth COM port for RIGHT_IMU
left_imu_port = 'COM8'   # Replace with your Bluetooth COM port for LEFT_IMU
baud_rate = 9600
data_timeout = 1  # Timeout for serial data

# Globals
right_serial = None
left_serial = None
gyro_taL = 0  # Placeholder for real-time GYRt_taL data
gyro_taL_history = []  # Store values for plotting
recording = False
recorded_data = {"right": defaultdict(list), "left": defaultdict(list)}

# Base paths for saving data
raw_data_dir = r"D:\cap2\bio_video\flask-gaitAnalysis\new_data"
interpolated_data_dir = r"D:\cap2\bio_video\flask-gaitAnalysis\interpolated_data"
combined_data_dir = r"D:\cap2\bio_video\flask-gaitAnalysis\combined_data"
os.makedirs(interpolated_data_dir, exist_ok=True)
os.makedirs(combined_data_dir, exist_ok=True)


app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///users.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)


class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(100), unique=True)
    password = db.Column(db.String(100))

    def __init__(self, username, password):
        self.username = username
        self.password = password


@app.route('/', methods=['GET'])
def index():
    if session.get('logged_in'):
        return render_template('home.html')
    else:
        return render_template('index.html', message="Welcome to Human Gait Recognition.")

## IMU Sensor data collection

# Connect to IMUs
@app.route('/connect_imus')
def connect_imus():
    global right_serial, left_serial
    try:
        right_serial = serial.Serial(right_imu_port, baud_rate, timeout=1)
        left_serial = serial.Serial(left_imu_port, baud_rate, timeout=1)
        return jsonify({"message": "IMUs Connected Successfully!"})
    except serial.SerialException as e:
        return jsonify({"message": f"Error Connecting to IMUs: {e}"})


# Fetch Sensor Data in Real Time
@app.route('/get_sensor_data')
def get_sensor_data():
    global gyro_taL
    return jsonify({"gyro_taL": gyro_taL})


# Provide Data for Plotting
@app.route('/get_plot_data')
def get_plot_data():
    return jsonify({"data": gyro_taL_history[-100:]})  # Send last 100 values

# Read Sensor Data Continuously (Every 0.1s) Without Printing Errors
def read_sensor_data():
    global gyro_taL, gyro_taL_history
    while True:
        try:
            if left_serial and left_serial.in_waiting > 0:
                left_data = left_serial.readline().decode().strip()
                left_parts = left_data.split(",")

                if len(left_parts) >= 7:
                    gyro_taL = float(left_parts[6])  # GYRy_taLValue
                    gyro_taL_history.append(gyro_taL)  # Store for plotting

                    # Keep only last 500 values for memory efficiency
                    if len(gyro_taL_history) > 500:
                        gyro_taL_history.pop(0)

        except (serial.SerialException, ValueError, UnicodeDecodeError):
            pass  # Avoid printing errors

        time.sleep(0.1)  # Update every 0.1s

# Start Background Thread for Reading Sensor Data
threading.Thread(target=read_sensor_data, daemon=True).start()

def record_sensor_data(folder_path, speed):
    global recorded_data, recording
    filename_taR = os.path.join(folder_path, f"taR_{speed}.csv")
    filename_taL = os.path.join(folder_path, f"taL_{speed}.csv")

    with open(filename_taR, "w", newline='') as file_taR, \
         open(filename_taL, "w", newline='') as file_taL:

        header = ["Timestamp (s)", "Sensor ID", "Accel X (m/s^2)", "Accel Y (m/s^2)", "Accel Z (m/s^2)",
                  "Gyro X (rad/s)", "Gyro Y (rad/s)", "Gyro Z (rad/s)"]
        csv_writer_taR = csv.writer(file_taR)
        csv_writer_taL = csv.writer(file_taL)
        csv_writer_taR.writerow(header)
        csv_writer_taL.writerow(header)
        while recording:
            try:
                right_data = right_serial.readline().decode().strip()
                left_data = left_serial.readline().decode().strip()
                if right_data:
                    right_parts = right_data.split(",")
                    if len(right_parts) == 8:
                        sensor_id = right_parts[1]
                        if sensor_id == '1':
                            csv_writer_taR.writerow(right_parts)
                            recorded_data["right"]["timestamp"].append(float(right_parts[0]))
                            recorded_data["right"]["gyro_x"].append(float(right_parts[5]))
                            recorded_data["right"]["gyro_y"].append(float(right_parts[6]))

                if left_data:
                    left_parts = left_data.split(",")
                    if len(left_parts) == 8:
                        sensor_id = left_parts[1]
                        if sensor_id == '1':
                            csv_writer_taL.writerow(left_parts)
                            recorded_data["left"]["timestamp"].append(float(left_parts[0]))
                            recorded_data["left"]["gyro_x"].append(float(left_parts[5]))
                            recorded_data["left"]["gyro_y"].append(float(left_parts[6]))

            except Exception as e:
                print(f"Error reading data: {e}")
            time.sleep(0.1)


@app.route('/register/', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        try:
            db.session.add(User(username=request.form['username'], password=request.form['password']))
            db.session.commit()
            return redirect(url_for('login'))
        except:
            return render_template('index.html', message="User Already Exists")
    else:
        return render_template('register.html')


@app.route('/login/', methods=['GET', 'POST'])
def login():
    if request.method == 'GET':
        return render_template('login.html')
    else:
        u = request.form['username']
        p = request.form['password']
        data = User.query.filter_by(username=u, password=p).first()
        if data is not None:
            session['logged_in'] = True
            return redirect(url_for('index'))
        return render_template('index.html', message="Incorrect Details")


# OpenCV Video Capture for iVCam (Usually it is device index 1)
camera = cv2.VideoCapture(1)  # If not working, try cv2.VideoCapture(0)

# Function to capture video frame-by-frame
def generate_frames():
    while True:
        success, frame = camera.read()
        if not success:
            break
        else:
            _, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


# Function to create the required folder structure
def create_folder(name):
    folder_path = os.path.join(raw_data_dir, name)
    os.makedirs(folder_path, exist_ok=True)
    return folder_path


@app.route('/logout', methods=['GET', 'POST'])
def logout():
    session['logged_in'] = False
    return redirect(url_for('index'))

# Route for streaming the video feed
@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

# Video Recording Variables
video_writer = None


# Start Recording
@app.route('/start_recording', methods=['POST'])
def start_recording():
    global video_writer, recording
    # Get data from frontend (name, age, gender)
    print("Hello")
    data = request.get_json()
    name = data.get("name")
    age = data.get("age")
    gender = data.get("gender")

    if right_serial is None or left_serial is None:
        return jsonify({"message": "Error: Connect to IMUs first!"})

    folder_path = create_folder(name)

    if not recording:
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        video_filename = os.path.join(folder_path, f"{name}_video.avi")
        video_writer = cv2.VideoWriter(video_filename, fourcc, 20.0, (640, 480))
        recording = True
        threading.Thread(target=record_video).start()
        # Start sensor data recording in a new thread
        threading.Thread(target=record_sensor_data, args=(folder_path, "session"), daemon=True).start()

        return {"message": "Both the recordings have started!"}
    else:
        return {"message": "Already recording!"}

# Stop Recording


@app.route('/start_verification', methods=['POST'])
def start_verification():
    global video_writer, recording

    print("Verification Started")

    if right_serial is None or left_serial is None:
        return jsonify({"message": "Error: Connect to IMUs first!"})

    folder_path = create_folder("verification_session")  # Save verification data in a fixed folder

    if not recording:
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        video_filename = os.path.join(folder_path, "verification_video.avi")
        video_writer = cv2.VideoWriter(video_filename, fourcc, 20.0, (640, 480))
        recording = True

        # Start video recording
        threading.Thread(target=record_video).start()

        # Start sensor data recording in a new thread
        threading.Thread(target=record_sensor_data, args=(folder_path, "verification_session"), daemon=True).start()

        return jsonify({"message": "The recording started!"})
    else:
        return jsonify({"message": "Already verifying!"})


@app.route('/stop_recording', methods=['POST'])
def stop_recording():
    global recording, video_writer
    if recording:
        recording = False
    if video_writer:
        video_writer.release()
        video_writer = None
    try:
        data = request.get_json()
        name = data.get("name")
        if name:
            process_sensor_data(name)  # Call function with name
            return jsonify({"message": f"Recording Stopped! The data has been processed {name}. Ready to onboard the user."})
        else:
            return jsonify({"message": "Recording Stopped! But no name provided for data processing."})
    except Exception as e:
        return jsonify({"message": f"Recording Stopped! But error in data processing: {str(e)}"})


@app.route('/stop_verification', methods=['POST'])
def stop_verification():
    global recording, video_writer

    if recording:
        recording = False

    if video_writer:
        video_writer.release()
        video_writer = None

    try:
        process_sensor_data("verification_session")  # Process data for verification
        return jsonify({"message": "Recording Stopped! The data has been processed. Ready to verify the user."})
    except Exception as e:
        return jsonify({"message": f"Verification Stopped! But error in data processing: {str(e)}"})


# Function to Record Video Frames
def record_video():
    global recording, video_writer
    while recording:
        success, frame = camera.read()
        if success and video_writer is not None:
            video_writer.write(frame)
    video_writer.release()


def convert_avi_to_mp4(infile, outfile):
    """
    Convert an AVI video to MP4 using FFmpeg.
    """
    print(f"Converting {infile} → {outfile}")

    command = [
        "ffmpeg",
        "-i", infile,  # Input file
        "-c:v", "copy",  # Copy video codec (no re-encoding)
        "-c:a", "copy",  # Copy audio codec
        "-y",  # Overwrite if file exists
        outfile
    ]

    try:
        subprocess.run(command, check=True)
        print(f"Conversion successful: {outfile}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"FFmpeg conversion failed: {e}")
        return False

@app.route('/train_model', methods=['POST'])
def train_model_route():
    """
    Receives user details, converts video to MP4, and calls the onboarding function.
    """
    data = request.get_json()
    name = data.get("name")
    age = data.get("age")
    gender = data.get("gender")

    if not name or not age or not gender:
        return jsonify({"message": "Error: Missing Name, Age, or Gender!"})

    # Construct file paths
    folder_path = os.path.join(raw_data_dir, name)
    avi_video_path = os.path.join(folder_path, f"{name}_video.avi")
    mp4_video_path = os.path.join(folder_path, f"{name}_video.mp4")

    # Convert AVI to MP4 first
    if not os.path.exists(avi_video_path):
        return jsonify({"message": "Error: AVI video file not found!"})

    if convert_avi_to_mp4(avi_video_path, mp4_video_path):
        # Call onboard function with MP4 file
        print("converting video")
        result_message = onboard_user(mp4_video_path, name, age, gender)

    return jsonify({"message": f"Onboarding Done for {name}!"})



@app.route('/verify_user', methods=['POST'])
def verify_user_route():
    """
    Calls the `verify_user()` function from `verify_user.py` and returns results.
    """
    try:
        # ✅ Define the video path (Change as needed)
        verification_video_path_in = os.path.join(raw_data_dir, "verification_session", "verification_video.avi")
        verification_video_path_out = os.path.join(raw_data_dir, "verification_session", "verification_video.mp4")
        if not os.path.exists(verification_video_path_in):
            return jsonify({"message": "Verification video not found!"})

        if convert_avi_to_mp4(verification_video_path_in,verification_video_path_out):
            print("Doing verification now")
            output_message, scores_table = verify_person(verification_video_path_out)

        scores_json = scores_table.to_json(orient="records")

        return jsonify({"message": output_message, "scores": scores_json})

    except Exception as e:
        return jsonify({"message": f"Error in verification: {str(e)}", "scores": "[]"})



@app.route('/register_user', methods=['GET', 'POST'])
def register_user():
    if request.method == 'POST':
        name = request.form['name']
        age = request.form['age']
        gender = request.form['gender']
        # Save user data (Modify this to save in database)
        print(f"New User Registered: {name}, {age}, {gender}")
        return redirect(url_for('register_user'))

    return render_template('register_user.html')

@app.route('/verify_user', methods=['GET', 'POST'])
def verify_user():
    if request.method == 'POST':
        # Handle user verification logic
        return redirect(url_for('home'))
    return render_template('verify_user.html')



@app.route('/analyze', methods=['POST'])
def analyze():
    """
    Perform sensor and video analysis and return results.
    """
    try:
        # Get file paths from request
        verification_session = r"combined_data\verification_session.csv"
        verification_video = r"new_data\verification_session\verification_video.mp4"

        if not verification_session or not verification_video:
            return jsonify({"message": "Error: Missing input files!"})

        # Process sensor and video data
        right_plot, left_plot, sensor_gait_table, sensor_step_count = process_gait(verification_session)
        video_step_count, output_video_path, video_gait_table, \
            video_plot_right, video_plot_left, stance_plot = process_video(verification_video)

        # Return all results to be displayed in the frontend
        return jsonify({
            "message": "Analysis Completed Successfully!",
            "output_video_path": output_video_path,
            "sensor_step_count": sensor_step_count,
            "video_step_count": video_step_count,
            "sensor_gait_table": sensor_gait_table.to_html(index=False),
            "video_gait_table": video_gait_table.to_html(index=False),
            "video_plot_right": video_plot_right,
            "video_plot_left": video_plot_left,
            "sensor_plot": right_plot,
            "stance_plot": stance_plot
        })

    except Exception as e:
        return jsonify({"message": f"Error during analysis: {str(e)}"})


@app.route('/analysis_page')
def analysis_page():
    """
    Render the analysis page.
    """
    return render_template("analysis.html")


if __name__ == '__main__':
    app.secret_key = "ThisIsNotASecret:p"
    # Fix: Create database tables inside an application context
    with app.app_context():
        db.create_all()
    app.run(debug=True)


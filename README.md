Gait Analysis Application

Overview

This application performs gait analysis using sensor and video models. It extracts and analyzes gait features to provide insights into user movement patterns.

Features:

Sensor-based gait recognition

Video-based gait analysis

Multi-modal fusion for improved accuracy

User registration and verification

Prerequisites:

Ensure you have the following installed:

Python 3.8+

TensorFlow 2.9

Protobuf 3.20 (for TensorFlow)


Virtual environment (recommended)

Clone the repository:

git clone https://github.com/zaina1864/GaitApp.git
cd GaitApp

Set up Virtual Environment:

python -m venv venv
source venv/bin/activate
pip install -r requirements.txt 



Running the Application:

Activate the environment and run the app:

source venv/bin/activate 
python app.py


Testing the Application:

Register a User

Open the application in your browser.

Navigate to the registration page.

Provide necessary details and submit.

Verify a User

Log in using registered credentials.

The system will check and verify the user.

Analyze Gait Data

Upload sensor or video data.

The system will process and display gait analysis results.

Troubleshooting

If you encounter Protobuf version conflicts, ensure you activate the correct environment before running the application.



Contributing

Feel free to fork this repository and submit pull requests for improvements.

License

This project is licensed under the MIT License.

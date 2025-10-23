# AI Exercise Coach & Human Pose Estimation App

This project leverages computer vision to perform real-time human pose estimation. It uses a pre-trained OpenPose model (with a MobileNet backbone) within OpenCV's DNN module to identify and track 18 human body keypoints.

This repository contains two distinct Streamlit applications:
1.  `estimation_app.py`: A **Basic Pose Viewer** that allows users to upload an image or video to see the detected pose skeleton.
2.  `exercise_trackapp.py`: An **Advanced AI Exercise Coach** that uses the pose data to track specific exercises, count repetitions, and provide real-time form feedback via a webcam or video file.

---

## 🚀 Features

### 1. Basic Pose Viewer (`estimation_app.py`)
* **Image & Video Support:** Upload static images (.jpg, .png) or video files (.mp4, .avi) to see the detected pose.
* **Confidence Slider:** Interactively adjust the confidence threshold for keypoint detection.
* **Video Seek Bar:** A slider allows you to choose which frame to start video processing from, so you can skip to the action.

### 2. AI Exercise Coach (`exercise_trackapp.py`)
This app includes all features from the basic viewer, plus:
* **Live Camera Feed:** Use your webcam for real-time analysis.
* **Exercise Selection:** A dropdown menu to select the exercise you want to track (currently supports Pushups and Squats).
* **Repetition Counting:**
    * **Pushup Counter:** Tracks elbow angle (e.g., > 160° for "up", < 90° for "down") to count full reps.
    * **Squat Counter:** Tracks knee angle and hip position relative to the knee to count valid reps.
* **Real-time Form Feedback:**
    * **Pushups:** Checks for a straight back by monitoring hip-to-shoulder alignment.
    * **Squats:** Checks for proper depth (hip below knee) and a straight back (prevents leaning over too far).
* **Visual Cues:**
    * Key joints are color-coded in real-time (**Green** for good form, **Red** for poor form).
    * A celebration (`st.balloons()`) fires on each successful repetition.
* **Error Handling:** Displays a "Body not in view" message if key points aren't detected for the selected exercise.

---

## 🛠️ Technology Stack

* **Python 3.10 / 3.11**
* **Streamlit:** For the web application UI and dashboard.
* **OpenCV (`cv2`):** For all image/video processing and DNN model inference.
* **TensorFlow Model:** A pre-trained OpenPose model (`graph_opt.pb`) with a MobileNet backbone.
* **Numpy:** For numerical operations and image manipulation.
* **Pillow (PIL):** For handling image uploads in Streamlit.

---

## ⚙️ Installation & Setup

Follow these steps to run the project locally.

### 1. Clone the Repository
Open your terminal and clone the repository:
```bash
git clone [https://github.com/Haswanthobbina/human-pose-identifier-.git](https://github.com/Haswanthobbina/human-pose-identifier-.git)
cd human-pose-identifier-
```
🏃‍♂️ How to Run the Apps
This repository contains two different apps. You can run either one.

To run the Simple Image/Video Viewer:
Bash

streamlit run estimation_app.py
To run the Advanced AI Exercise Coach:
Bash

streamlit run exercise_trackapp.py
After running the command, Streamlit will open the application in your web browser. Select your desired input source, exercise, and options in the sidebar to begin.

📂 Project Structure
.
├── exercise_trackapp.py    # The advanced AI coach app
├── estimation_app.py       # The simple image/video viewer app
├── graph_opt.pb            # The TensorFlow model (handled by Git LFS)
├── requirements.txt        # Python library dependencies
├── stand.jpg               # A demo image for testing
└── README.md               # This file

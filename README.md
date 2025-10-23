# Human Pose Identifier and Exercise Tracker

A real-time human pose estimation and exercise tracking system using deep learning and computer vision. This application leverages **MediaPipe** for pose detection and **OpenCV** for video processing to track and count exercise repetitions.

## Features

- **Real-time Pose Estimation**: Uses MediaPipe's deep learning models for accurate human pose detection
- **Exercise Tracking**: Automatically counts repetitions for various exercises
- **Multiple Exercise Support**:
  - Squats
  - Push-ups
  - Bicep Curls
- **Angle Calculation**: Real-time joint angle measurements for form assessment
- **Video and Webcam Support**: Works with live webcam feed or pre-recorded videos
- **Visual Feedback**: Displays pose landmarks, rep counter, and exercise stage

## Technology Stack

- **Deep Learning**: MediaPipe Pose (Google's ML solution for pose estimation)
- **Computer Vision**: OpenCV for video processing and visualization
- **Python**: Core programming language
- **NumPy**: Mathematical operations and angle calculations

## Installation

1. Clone the repository:
```bash
git clone https://github.com/Haswanthobbina/human-pose-identifier-.git
cd human-pose-identifier-
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Basic Usage (Webcam)

Track squats using your webcam:
```bash
python main.py
```

### Specify Exercise Type

Track push-ups:
```bash
python main.py --exercise pushup
```

Track bicep curls:
```bash
python main.py --exercise bicep-curl
```

### Use Video File

Track exercises from a video file:
```bash
python main.py --source path/to/video.mp4 --exercise squat
```

### Adjust Detection Confidence

Set custom confidence threshold (0.0 to 1.0):
```bash
python main.py --confidence 0.7
```

## Command Line Arguments

- `--source`: Video source (default: '0' for webcam, or path to video file)
- `--exercise`: Exercise type to track (choices: 'squat', 'pushup', 'bicep-curl', default: 'squat')
- `--confidence`: Minimum detection confidence (default: 0.5)

## Controls

- **Press 'q'**: Quit the application
- **Press 'r'**: Reset the repetition counter

## How It Works

### Pose Estimation
The application uses MediaPipe's Pose solution, which employs deep learning models to detect 33 3D body landmarks. The model is trained on diverse datasets and provides robust pose detection in real-time.

### Exercise Tracking
The exercise tracker calculates angles between specific body joints:

- **Squats**: Measures the angle at the knee joint (hip-knee-ankle)
  - Down position: angle < 90°
  - Up position: angle > 160°

- **Push-ups**: Measures the angle at the elbow joint (shoulder-elbow-wrist)
  - Down position: angle < 90°
  - Up position: angle > 160°

- **Bicep Curls**: Measures the angle at the elbow joint (shoulder-elbow-wrist)
  - Down position: angle > 160°
  - Up position: angle < 30°

A repetition is counted when the exercise transitions from the starting position through the end position and back.

## Project Structure

```
human-pose-identifier-/
├── main.py                 # Main application entry point
├── pose_estimator.py       # Pose estimation module using MediaPipe
├── exercise_tracker.py     # Exercise tracking and counting logic
├── angle_utils.py         # Angle calculation utilities
├── requirements.txt       # Python dependencies
└── README.md             # Documentation
```

## Requirements

- Python 3.7+
- OpenCV (opencv-python)
- MediaPipe
- NumPy

See `requirements.txt` for specific versions.

## Machine Learning & Deep Learning Components

### MediaPipe Pose
- **Architecture**: BlazePose, a lightweight convolutional neural network
- **Inference**: Real-time pose detection at 30+ FPS on modern hardware
- **Landmarks**: 33 3D body landmarks with x, y, z coordinates and visibility scores
- **Training**: Pre-trained on large-scale datasets with diverse poses and body types

### Angle Calculation
Uses trigonometric calculations and vector mathematics to compute joint angles from detected landmarks for exercise form analysis.

## Future Enhancements

- Support for more exercise types (lunges, jumping jacks, etc.)
- Form correction feedback using pose analysis
- Exercise session tracking and statistics
- Mobile application support
- Multi-person pose tracking
- Exercise recommendation system

## License

See LICENSE file for details.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Acknowledgments

- Google's MediaPipe team for the pose estimation solution
- OpenCV community for computer vision tools
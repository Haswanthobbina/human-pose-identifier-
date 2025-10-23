# Human Pose Identifier

A Python-based human pose detection and tracking system using MediaPipe and OpenCV. This project provides real-time pose estimation capabilities for images, videos, and webcam streams.

## Features

- Real-time pose detection using MediaPipe
- Support for images, videos, and webcam input
- Extract 33 pose landmarks with 3D coordinates
- Easy-to-use Python API
- Configurable detection and tracking confidence thresholds

## Requirements

- Python 3.7 or higher
- OpenCV
- MediaPipe
- NumPy

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

### Webcam Detection

Run pose detection on your webcam:
```bash
python -m src.pose_detector
```

Press 'q' to quit the webcam view.

### Image Detection

Detect pose in a single image:
```bash
python example.py path/to/your/image.jpg
```

### Using as a Library

```python
from src.pose_detector import PoseDetector
import cv2

# Initialize detector
detector = PoseDetector()

# Read an image
image = cv2.imread('image.jpg')

# Detect pose
result_image, results = detector.detect_pose(image)

# Get landmarks
landmarks = detector.get_landmarks(results)

# Cleanup
detector.close()
```

## Project Structure

```
human-pose-identifier-/
├── src/
│   ├── __init__.py
│   └── pose_detector.py    # Main pose detection module
├── example.py              # Example usage script
├── requirements.txt        # Python dependencies
├── README.md              # This file
├── LICENSE                # GPL-3.0 License
└── .gitignore            # Git ignore rules
```

## Pose Landmarks

MediaPipe Pose detects 33 body landmarks including:
- Face (nose, eyes, ears, mouth)
- Upper body (shoulders, elbows, wrists)
- Torso (hips)
- Lower body (knees, ankles, feet)

Each landmark provides:
- x, y coordinates (normalized to [0.0, 1.0])
- z coordinate (depth)
- visibility score

## License

This project is licensed under the GNU General Public License v3.0 - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- [MediaPipe](https://google.github.io/mediapipe/) by Google for pose detection
- [OpenCV](https://opencv.org/) for computer vision utilities
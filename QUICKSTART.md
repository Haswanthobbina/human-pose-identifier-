# Quick Start Guide

## Human Pose Identifier and Exercise Tracker

This guide will help you get started with the Human Pose Identifier and Exercise Tracker in just a few minutes.

## Prerequisites

- Python 3.7 or higher
- Webcam (optional, for live tracking)
- Video file (optional, for video analysis)

## Installation Steps

### 1. Clone the Repository

```bash
git clone https://github.com/Haswanthobbina/human-pose-identifier-.git
cd human-pose-identifier-
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

This will install:
- OpenCV (for video processing)
- MediaPipe (for pose estimation using deep learning)
- NumPy (for mathematical operations)

### 3. Run the Demo

To see how the angle calculations work without needing a camera:

```bash
python demo.py
```

This will demonstrate:
- Angle calculations for different exercises
- Distance calculations
- Exercise counting logic

### 4. Run Unit Tests

To verify everything is working correctly:

```bash
python -m unittest test_exercise_tracker -v
```

Expected output: All 11 tests should pass.

## Using the Exercise Tracker

### Track Squats with Webcam

```bash
python main.py --exercise squat
```

### Track Push-ups with Webcam

```bash
python main.py --exercise pushup
```

### Track Bicep Curls with Webcam

```bash
python main.py --exercise bicep-curl
```

### Analyze a Video File

```bash
python main.py --source path/to/your/video.mp4 --exercise squat
```

### Adjust Detection Confidence

If the tracking is too sensitive or not sensitive enough:

```bash
python main.py --exercise squat --confidence 0.7
```

Confidence values range from 0.0 to 1.0:
- Lower values (0.3-0.5): More detections, but may include false positives
- Higher values (0.7-0.9): Fewer but more accurate detections

## Controls During Exercise Tracking

When the application is running:

- **Press 'q'**: Quit the application
- **Press 'r'**: Reset the repetition counter

## What You'll See

The application displays:
1. **Live video feed** with pose landmarks drawn on your body
2. **Rep counter** showing the number of completed repetitions
3. **Current stage** (up/down) of the exercise
4. **Joint angle** in degrees for the tracked movement
5. **Exercise type** being tracked

## Exercise Form Guide

### Squats
- **Starting position**: Stand upright (knee angle ~180°)
- **Squat down**: Bend knees until angle < 90°
- **Count**: One rep when you go down and back up

### Push-ups
- **Starting position**: Arms extended (elbow angle ~180°)
- **Lower down**: Bend elbows until angle < 90°
- **Count**: One rep when you go down and back up

### Bicep Curls
- **Starting position**: Arms extended (elbow angle ~180°)
- **Curl up**: Bend elbows until angle < 30°
- **Count**: One rep when you curl up and back down

## Troubleshooting

### Camera Not Working
- Make sure no other application is using the camera
- Try specifying the camera explicitly: `--source 1` or `--source 2`
- Check camera permissions on your system

### Low Frame Rate
- Close other applications to free up resources
- Try reducing the confidence threshold: `--confidence 0.3`
- Ensure good lighting in your room

### Not Detecting Poses
- Ensure you're within camera view (2-3 meters away)
- Make sure there's adequate lighting
- Wear clothing that contrasts with the background
- Try adjusting confidence: `--confidence 0.5`

### Counter Not Incrementing
- Ensure you're completing the full range of motion
- Check that the tracked joint is clearly visible to the camera
- Avoid wearing loose clothing that obscures joints
- Position yourself so the camera has a side or front view

## Technology Used

- **MediaPipe Pose**: Google's machine learning solution for human pose estimation
  - Uses BlazePose neural network architecture
  - Detects 33 body landmarks in real-time
  - Trained on diverse datasets for robust performance

- **OpenCV**: Computer vision library for video processing
  - Captures video from webcam or files
  - Renders pose landmarks and overlays
  - Handles image transformations

- **NumPy**: Numerical computing library
  - Performs angle calculations using trigonometry
  - Computes distances between points
  - Vector mathematics for pose analysis

## Next Steps

- Try different exercises and video sources
- Experiment with confidence thresholds
- Share your results and provide feedback
- Contribute new exercise types

## Support

For issues or questions:
- Open an issue on GitHub
- Review the README.md for detailed documentation
- Check the source code comments for implementation details

Happy tracking! 💪

#!/usr/bin/env python3
"""
Example script demonstrating human pose detection on an image.
"""

import cv2
import sys
from src.pose_detector import PoseDetector


def detect_pose_in_image(image_path):
    """
    Detect pose in a single image.
    
    Args:
        image_path (str): Path to the input image
    """
    # Initialize detector
    detector = PoseDetector(static_image_mode=True)
    
    # Read image
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Could not read image from {image_path}")
        return
    
    # Detect pose
    result_image, results = detector.detect_pose(image)
    
    # Get landmarks
    landmarks = detector.get_landmarks(results)
    
    if landmarks:
        print(f"Detected {len(landmarks)} pose landmarks")
        print("Sample landmarks (first 5):")
        for i, landmark in enumerate(landmarks[:5]):
            print(f"  Landmark {i}: x={landmark['x']:.3f}, y={landmark['y']:.3f}, "
                  f"z={landmark['z']:.3f}, visibility={landmark['visibility']:.3f}")
    else:
        print("No pose detected in the image")
    
    # Display result
    cv2.imshow('Pose Detection Result', result_image)
    print("Press any key to close the window...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    # Cleanup
    detector.close()


def main():
    """Main function."""
    if len(sys.argv) < 2:
        print("Usage: python example.py <image_path>")
        print("Example: python example.py sample_image.jpg")
        print("\nTo use webcam instead, run:")
        print("  python -m src.pose_detector")
        sys.exit(1)
    
    image_path = sys.argv[1]
    detect_pose_in_image(image_path)


if __name__ == "__main__":
    main()

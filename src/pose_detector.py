"""
Human Pose Identifier
A Python module for detecting and tracking human poses using MediaPipe.
"""

import cv2
import mediapipe as mp
import numpy as np


class PoseDetector:
    """
    A class to detect and track human poses in images and videos.
    """
    
    def __init__(self, static_image_mode=False, model_complexity=1, 
                 min_detection_confidence=0.5, min_tracking_confidence=0.5):
        """
        Initialize the PoseDetector.
        
        Args:
            static_image_mode (bool): Whether to treat input as static images
            model_complexity (int): Complexity of the pose landmark model (0, 1, or 2)
            min_detection_confidence (float): Minimum confidence for detection
            min_tracking_confidence (float): Minimum confidence for tracking
        """
        self.mp_pose = mp.solutions.pose
        self.mp_drawing = mp.solutions.drawing_utils
        self.pose = self.mp_pose.Pose(
            static_image_mode=static_image_mode,
            model_complexity=model_complexity,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence
        )
        
    def detect_pose(self, image, draw=True):
        """
        Detect pose landmarks in an image.
        
        Args:
            image: Input image (BGR format)
            draw (bool): Whether to draw landmarks on the image
            
        Returns:
            tuple: (image with landmarks, pose results)
        """
        # Convert BGR to RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Process the image
        results = self.pose.process(image_rgb)
        
        # Draw landmarks if requested
        if draw and results.pose_landmarks:
            self.mp_drawing.draw_landmarks(
                image, 
                results.pose_landmarks, 
                self.mp_pose.POSE_CONNECTIONS
            )
        
        return image, results
    
    def get_landmarks(self, results):
        """
        Extract landmark coordinates from results.
        
        Args:
            results: Pose detection results
            
        Returns:
            list: List of landmark coordinates (x, y, z, visibility)
        """
        if results.pose_landmarks:
            landmarks = []
            for landmark in results.pose_landmarks.landmark:
                landmarks.append({
                    'x': landmark.x,
                    'y': landmark.y,
                    'z': landmark.z,
                    'visibility': landmark.visibility
                })
            return landmarks
        return None
    
    def close(self):
        """Close the pose detector."""
        self.pose.close()


def main():
    """
    Main function to demonstrate pose detection on webcam.
    """
    # Initialize pose detector
    detector = PoseDetector()
    
    # Open webcam
    cap = cv2.VideoCapture(0)
    
    print("Starting pose detection. Press 'q' to quit.")
    
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            print("Failed to read from webcam")
            break
        
        # Detect pose
        frame, results = detector.detect_pose(frame)
        
        # Display the frame
        cv2.imshow('Human Pose Detection', frame)
        
        # Exit on 'q' press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # Cleanup
    cap.release()
    cv2.destroyAllWindows()
    detector.close()


if __name__ == "__main__":
    main()

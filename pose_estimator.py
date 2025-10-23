"""
Pose Estimator Module
Uses MediaPipe for real-time human pose estimation
"""

import cv2
import mediapipe as mp
import numpy as np


class PoseEstimator:
    """
    Estimates human pose using MediaPipe Pose solution
    """
    
    def __init__(self, min_detection_confidence=0.5, min_tracking_confidence=0.5):
        """
        Initialize the pose estimator
        
        Args:
            min_detection_confidence: Minimum confidence for pose detection
            min_tracking_confidence: Minimum confidence for pose tracking
        """
        self.mp_pose = mp.solutions.pose
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        
        self.pose = self.mp_pose.Pose(
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence
        )
        
    def detect_pose(self, frame):
        """
        Detect pose in a frame
        
        Args:
            frame: Input image frame (BGR format)
            
        Returns:
            results: MediaPipe pose results
            frame_rgb: RGB converted frame
        """
        # Convert BGR to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Process the frame
        results = self.pose.process(frame_rgb)
        
        return results, frame_rgb
    
    def draw_landmarks(self, frame, results):
        """
        Draw pose landmarks on the frame
        
        Args:
            frame: Input image frame
            results: MediaPipe pose results
            
        Returns:
            frame: Frame with drawn landmarks
        """
        if results.pose_landmarks:
            self.mp_drawing.draw_landmarks(
                frame,
                results.pose_landmarks,
                self.mp_pose.POSE_CONNECTIONS,
                landmark_drawing_spec=self.mp_drawing_styles.get_default_pose_landmarks_style()
            )
        return frame
    
    def get_landmark_coordinates(self, results, landmark_id, frame_width, frame_height):
        """
        Get pixel coordinates of a specific landmark
        
        Args:
            results: MediaPipe pose results
            landmark_id: ID of the landmark
            frame_width: Width of the frame
            frame_height: Height of the frame
            
        Returns:
            tuple: (x, y) coordinates or None if not detected
        """
        if results.pose_landmarks:
            landmark = results.pose_landmarks.landmark[landmark_id]
            x = int(landmark.x * frame_width)
            y = int(landmark.y * frame_height)
            return (x, y)
        return None
    
    def close(self):
        """Close the pose estimator"""
        self.pose.close()

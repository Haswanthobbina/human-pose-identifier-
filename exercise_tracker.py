"""
Exercise Tracker Module
Tracks and counts various exercises based on pose estimation
"""

import cv2
from angle_utils import calculate_angle
import mediapipe as mp


class ExerciseTracker:
    """
    Tracks exercises and counts repetitions
    """
    
    def __init__(self):
        """Initialize the exercise tracker"""
        self.counter = 0
        self.stage = None
        self.mp_pose = mp.solutions.pose
        
    def reset_counter(self):
        """Reset the exercise counter"""
        self.counter = 0
        self.stage = None
    
    def track_squat(self, results, frame_width, frame_height):
        """
        Track squat exercise
        
        Args:
            results: MediaPipe pose results
            frame_width: Width of the frame
            frame_height: Height of the frame
            
        Returns:
            dict: Exercise tracking data
        """
        if not results.pose_landmarks:
            return None
        
        # Get landmarks
        landmarks = results.pose_landmarks.landmark
        
        # Get coordinates for left leg
        hip = (int(landmarks[self.mp_pose.PoseLandmark.LEFT_HIP.value].x * frame_width),
               int(landmarks[self.mp_pose.PoseLandmark.LEFT_HIP.value].y * frame_height))
        knee = (int(landmarks[self.mp_pose.PoseLandmark.LEFT_KNEE.value].x * frame_width),
                int(landmarks[self.mp_pose.PoseLandmark.LEFT_KNEE.value].y * frame_height))
        ankle = (int(landmarks[self.mp_pose.PoseLandmark.LEFT_ANKLE.value].x * frame_width),
                 int(landmarks[self.mp_pose.PoseLandmark.LEFT_ANKLE.value].y * frame_height))
        
        # Calculate angle
        angle = calculate_angle(hip, knee, ankle)
        
        if angle is None:
            return None
        
        # Squat logic: down when angle < 90, up when angle > 160
        if angle > 160:
            self.stage = "up"
        if angle < 90 and self.stage == "up":
            self.stage = "down"
            self.counter += 1
        
        return {
            'exercise': 'squat',
            'angle': angle,
            'counter': self.counter,
            'stage': self.stage
        }
    
    def track_pushup(self, results, frame_width, frame_height):
        """
        Track push-up exercise
        
        Args:
            results: MediaPipe pose results
            frame_width: Width of the frame
            frame_height: Height of the frame
            
        Returns:
            dict: Exercise tracking data
        """
        if not results.pose_landmarks:
            return None
        
        # Get landmarks
        landmarks = results.pose_landmarks.landmark
        
        # Get coordinates for left arm
        shoulder = (int(landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER.value].x * frame_width),
                    int(landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER.value].y * frame_height))
        elbow = (int(landmarks[self.mp_pose.PoseLandmark.LEFT_ELBOW.value].x * frame_width),
                 int(landmarks[self.mp_pose.PoseLandmark.LEFT_ELBOW.value].y * frame_height))
        wrist = (int(landmarks[self.mp_pose.PoseLandmark.LEFT_WRIST.value].x * frame_width),
                 int(landmarks[self.mp_pose.PoseLandmark.LEFT_WRIST.value].y * frame_height))
        
        # Calculate angle
        angle = calculate_angle(shoulder, elbow, wrist)
        
        if angle is None:
            return None
        
        # Push-up logic: down when angle < 90, up when angle > 160
        if angle > 160:
            self.stage = "up"
        if angle < 90 and self.stage == "up":
            self.stage = "down"
            self.counter += 1
        
        return {
            'exercise': 'push-up',
            'angle': angle,
            'counter': self.counter,
            'stage': self.stage
        }
    
    def track_bicep_curl(self, results, frame_width, frame_height):
        """
        Track bicep curl exercise
        
        Args:
            results: MediaPipe pose results
            frame_width: Width of the frame
            frame_height: Height of the frame
            
        Returns:
            dict: Exercise tracking data
        """
        if not results.pose_landmarks:
            return None
        
        # Get landmarks
        landmarks = results.pose_landmarks.landmark
        
        # Get coordinates for left arm
        shoulder = (int(landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER.value].x * frame_width),
                    int(landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER.value].y * frame_height))
        elbow = (int(landmarks[self.mp_pose.PoseLandmark.LEFT_ELBOW.value].x * frame_width),
                 int(landmarks[self.mp_pose.PoseLandmark.LEFT_ELBOW.value].y * frame_height))
        wrist = (int(landmarks[self.mp_pose.PoseLandmark.LEFT_WRIST.value].x * frame_width),
                 int(landmarks[self.mp_pose.PoseLandmark.LEFT_WRIST.value].y * frame_height))
        
        # Calculate angle
        angle = calculate_angle(shoulder, elbow, wrist)
        
        if angle is None:
            return None
        
        # Bicep curl logic: down when angle > 160, up when angle < 30
        if angle > 160:
            self.stage = "down"
        if angle < 30 and self.stage == "down":
            self.stage = "up"
            self.counter += 1
        
        return {
            'exercise': 'bicep-curl',
            'angle': angle,
            'counter': self.counter,
            'stage': self.stage
        }

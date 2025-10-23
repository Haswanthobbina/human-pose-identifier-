"""
Human Pose Identifier and Exercise Tracker
Main application for real-time pose estimation and exercise tracking
"""

import cv2
import argparse
from pose_estimator import PoseEstimator
from exercise_tracker import ExerciseTracker


def main():
    """Main application function"""
    parser = argparse.ArgumentParser(description='Human Pose Identifier and Exercise Tracker')
    parser.add_argument('--source', type=str, default='0', 
                        help='Video source: 0 for webcam or path to video file')
    parser.add_argument('--exercise', type=str, default='squat',
                        choices=['squat', 'pushup', 'bicep-curl'],
                        help='Type of exercise to track')
    parser.add_argument('--confidence', type=float, default=0.5,
                        help='Minimum detection confidence (0.0 to 1.0)')
    
    args = parser.parse_args()
    
    # Initialize video source
    if args.source.isdigit():
        cap = cv2.VideoCapture(int(args.source))
    else:
        cap = cv2.VideoCapture(args.source)
    
    if not cap.isOpened():
        print("Error: Could not open video source")
        return
    
    # Initialize pose estimator and exercise tracker
    pose_estimator = PoseEstimator(
        min_detection_confidence=args.confidence,
        min_tracking_confidence=args.confidence
    )
    exercise_tracker = ExerciseTracker()
    
    print(f"Starting exercise tracker for: {args.exercise}")
    print("Press 'q' to quit, 'r' to reset counter")
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("End of video or cannot read frame")
            break
        
        # Get frame dimensions
        frame_height, frame_width = frame.shape[:2]
        
        # Detect pose
        results, _ = pose_estimator.detect_pose(frame)
        
        # Draw landmarks
        frame = pose_estimator.draw_landmarks(frame, results)
        
        # Track exercise
        tracking_data = None
        if args.exercise == 'squat':
            tracking_data = exercise_tracker.track_squat(results, frame_width, frame_height)
        elif args.exercise == 'pushup':
            tracking_data = exercise_tracker.track_pushup(results, frame_width, frame_height)
        elif args.exercise == 'bicep-curl':
            tracking_data = exercise_tracker.track_bicep_curl(results, frame_width, frame_height)
        
        # Display tracking information
        if tracking_data:
            # Display counter
            cv2.putText(frame, f"Reps: {tracking_data['counter']}", 
                        (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 
                        1.5, (0, 255, 0), 3)
            
            # Display stage
            cv2.putText(frame, f"Stage: {tracking_data['stage']}", 
                        (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 
                        1, (255, 255, 255), 2)
            
            # Display angle
            cv2.putText(frame, f"Angle: {int(tracking_data['angle'])}", 
                        (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 
                        1, (255, 255, 255), 2)
        
        # Display exercise type
        cv2.putText(frame, f"Exercise: {args.exercise.upper()}", 
                    (10, frame_height - 20), cv2.FONT_HERSHEY_SIMPLEX, 
                    0.8, (255, 255, 0), 2)
        
        # Show frame
        cv2.imshow('Human Pose Identifier - Exercise Tracker', frame)
        
        # Handle key press
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('r'):
            exercise_tracker.reset_counter()
            print("Counter reset")
    
    # Cleanup
    cap.release()
    cv2.destroyAllWindows()
    pose_estimator.close()
    
    print(f"Final count: {exercise_tracker.counter} reps")


if __name__ == "__main__":
    main()

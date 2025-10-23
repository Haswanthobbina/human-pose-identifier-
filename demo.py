"""
Demo script to show the pose estimation and exercise tracking capabilities
This demonstrates the functionality without requiring a camera or video file
"""

from angle_utils import calculate_angle, calculate_distance


def demo_angle_calculation():
    """Demonstrate angle calculation for exercise tracking"""
    print("=" * 60)
    print("ANGLE CALCULATION DEMO")
    print("=" * 60)
    
    # Squat example - knee angle
    print("\n1. Squat Position Analysis:")
    print("   Calculating knee angle (hip-knee-ankle)")
    
    # Standing position
    hip = (100, 50)
    knee = (100, 150)
    ankle = (100, 250)
    standing_angle = calculate_angle(hip, knee, ankle)
    print(f"   Standing position: {standing_angle:.1f}° (should be ~180°)")
    
    # Squat down position
    hip = (100, 50)
    knee = (100, 150)
    ankle = (150, 200)
    squat_angle = calculate_angle(hip, knee, ankle)
    print(f"   Squat down position: {squat_angle:.1f}° (should be <90°)")
    
    # Push-up example - elbow angle
    print("\n2. Push-up Position Analysis:")
    print("   Calculating elbow angle (shoulder-elbow-wrist)")
    
    # Up position
    shoulder = (50, 100)
    elbow = (150, 100)
    wrist = (250, 100)
    up_angle = calculate_angle(shoulder, elbow, wrist)
    print(f"   Up position: {up_angle:.1f}° (should be ~180°)")
    
    # Down position
    shoulder = (50, 100)
    elbow = (150, 100)
    wrist = (150, 150)
    down_angle = calculate_angle(shoulder, elbow, wrist)
    print(f"   Down position: {down_angle:.1f}° (should be <90°)")
    
    # Bicep curl example - elbow angle
    print("\n3. Bicep Curl Position Analysis:")
    print("   Calculating elbow angle (shoulder-elbow-wrist)")
    
    # Down position (arm extended)
    shoulder = (100, 50)
    elbow = (100, 150)
    wrist = (100, 250)
    extended_angle = calculate_angle(shoulder, elbow, wrist)
    print(f"   Extended (down): {extended_angle:.1f}° (should be ~180°)")
    
    # Up position (arm curled)
    shoulder = (100, 50)
    elbow = (100, 150)
    wrist = (120, 80)
    curled_angle = calculate_angle(shoulder, elbow, wrist)
    print(f"   Curled (up): {curled_angle:.1f}° (should be <30°)")


def demo_distance_calculation():
    """Demonstrate distance calculation"""
    print("\n" + "=" * 60)
    print("DISTANCE CALCULATION DEMO")
    print("=" * 60)
    
    point1 = (0, 0)
    point2 = (3, 4)
    dist = calculate_distance(point1, point2)
    print(f"\nDistance between {point1} and {point2}: {dist:.1f} pixels")
    print(f"Expected: 5.0 (by Pythagorean theorem: sqrt(3^2 + 4^2))")


def demo_exercise_logic():
    """Demonstrate exercise counting logic"""
    print("\n" + "=" * 60)
    print("EXERCISE COUNTING LOGIC")
    print("=" * 60)
    
    print("\nSQUAT COUNTER LOGIC:")
    print("- Stage 'up' when knee angle > 160°")
    print("- Stage 'down' when knee angle < 90° (from 'up' stage)")
    print("- Count increments when transitioning from 'up' to 'down'")
    
    print("\nPUSH-UP COUNTER LOGIC:")
    print("- Stage 'up' when elbow angle > 160°")
    print("- Stage 'down' when elbow angle < 90° (from 'up' stage)")
    print("- Count increments when transitioning from 'up' to 'down'")
    
    print("\nBICEP CURL COUNTER LOGIC:")
    print("- Stage 'down' when elbow angle > 160°")
    print("- Stage 'up' when elbow angle < 30° (from 'down' stage)")
    print("- Count increments when transitioning from 'down' to 'up'")


def main():
    """Run all demos"""
    print("\n")
    print("*" * 60)
    print("HUMAN POSE IDENTIFIER & EXERCISE TRACKER - DEMO")
    print("*" * 60)
    
    demo_angle_calculation()
    demo_distance_calculation()
    demo_exercise_logic()
    
    print("\n" + "=" * 60)
    print("FEATURES")
    print("=" * 60)
    print("\n✓ Real-time pose estimation using MediaPipe")
    print("✓ Automatic exercise counting (Squats, Push-ups, Bicep Curls)")
    print("✓ Joint angle calculations for form assessment")
    print("✓ Video and webcam support")
    print("✓ Visual feedback with pose landmarks")
    
    print("\n" + "=" * 60)
    print("USAGE")
    print("=" * 60)
    print("\nTo run the exercise tracker:")
    print("  python main.py --exercise squat")
    print("  python main.py --exercise pushup")
    print("  python main.py --exercise bicep-curl")
    print("\nWith video file:")
    print("  python main.py --source video.mp4 --exercise squat")
    print("\nControls:")
    print("  'q' - Quit")
    print("  'r' - Reset counter")
    
    print("\n" + "*" * 60)
    print("Demo completed successfully!")
    print("*" * 60 + "\n")


if __name__ == "__main__":
    main()

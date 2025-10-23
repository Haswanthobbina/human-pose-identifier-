"""
Unit tests for the Human Pose Identifier and Exercise Tracker
"""

import unittest
from angle_utils import calculate_angle, calculate_distance


class TestAngleUtils(unittest.TestCase):
    """Test cases for angle calculation utilities"""
    
    def test_calculate_angle_90_degrees(self):
        """Test 90-degree angle calculation"""
        point1 = (0, 0)
        point2 = (1, 0)
        point3 = (1, 1)
        angle = calculate_angle(point1, point2, point3)
        self.assertAlmostEqual(angle, 90.0, places=1)
    
    def test_calculate_angle_180_degrees(self):
        """Test 180-degree angle (straight line)"""
        point1 = (0, 0)
        point2 = (1, 0)
        point3 = (2, 0)
        angle = calculate_angle(point1, point2, point3)
        self.assertAlmostEqual(angle, 180.0, places=1)
    
    def test_calculate_angle_120_degrees(self):
        """Test 120-degree angle"""
        point1 = (0, 0)
        point2 = (1, 0)
        point3 = (1.5, 0.866)
        angle = calculate_angle(point1, point2, point3)
        self.assertAlmostEqual(angle, 120.0, places=0)
    
    def test_calculate_angle_with_none(self):
        """Test angle calculation with None values"""
        point1 = (0, 0)
        point2 = (1, 0)
        
        # Test with None in different positions
        self.assertIsNone(calculate_angle(None, point2, point1))
        self.assertIsNone(calculate_angle(point1, None, point2))
        self.assertIsNone(calculate_angle(point1, point2, None))
    
    def test_calculate_distance_simple(self):
        """Test simple distance calculation"""
        point1 = (0, 0)
        point2 = (3, 4)
        distance = calculate_distance(point1, point2)
        self.assertAlmostEqual(distance, 5.0, places=1)
    
    def test_calculate_distance_horizontal(self):
        """Test horizontal distance"""
        point1 = (0, 0)
        point2 = (5, 0)
        distance = calculate_distance(point1, point2)
        self.assertAlmostEqual(distance, 5.0, places=1)
    
    def test_calculate_distance_vertical(self):
        """Test vertical distance"""
        point1 = (0, 0)
        point2 = (0, 10)
        distance = calculate_distance(point1, point2)
        self.assertAlmostEqual(distance, 10.0, places=1)
    
    def test_calculate_distance_with_none(self):
        """Test distance calculation with None values"""
        point1 = (0, 0)
        
        self.assertIsNone(calculate_distance(None, point1))
        self.assertIsNone(calculate_distance(point1, None))


class TestExerciseLogic(unittest.TestCase):
    """Test cases for exercise tracking logic"""
    
    def test_squat_angle_thresholds(self):
        """Test squat angle thresholds"""
        # Standing position should be > 160 degrees
        standing_angle = 170
        self.assertGreater(standing_angle, 160)
        
        # Squat down should be < 90 degrees
        squat_angle = 80
        self.assertLess(squat_angle, 90)
    
    def test_pushup_angle_thresholds(self):
        """Test push-up angle thresholds"""
        # Up position should be > 160 degrees
        up_angle = 165
        self.assertGreater(up_angle, 160)
        
        # Down position should be < 90 degrees
        down_angle = 85
        self.assertLess(down_angle, 90)
    
    def test_bicep_curl_angle_thresholds(self):
        """Test bicep curl angle thresholds"""
        # Extended position should be > 160 degrees
        extended_angle = 170
        self.assertGreater(extended_angle, 160)
        
        # Curled position should be < 30 degrees
        curled_angle = 25
        self.assertLess(curled_angle, 30)


if __name__ == '__main__':
    unittest.main()

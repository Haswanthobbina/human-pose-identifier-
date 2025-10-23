"""
Angle Calculation Utilities
Helper functions for calculating angles between body joints
"""

import numpy as np


def calculate_angle(point1, point2, point3):
    """
    Calculate angle between three points
    
    Args:
        point1: First point (x, y)
        point2: Middle point (vertex) (x, y)
        point3: Third point (x, y)
        
    Returns:
        angle: Angle in degrees
    """
    if point1 is None or point2 is None or point3 is None:
        return None
    
    # Convert points to numpy arrays
    a = np.array(point1)
    b = np.array(point2)
    c = np.array(point3)
    
    # Calculate vectors
    ba = a - b
    bc = c - b
    
    # Calculate angle using dot product
    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    
    # Ensure cosine_angle is within valid range [-1, 1]
    cosine_angle = np.clip(cosine_angle, -1.0, 1.0)
    
    angle = np.arccos(cosine_angle)
    
    # Convert to degrees
    angle_degrees = np.degrees(angle)
    
    return angle_degrees


def calculate_distance(point1, point2):
    """
    Calculate Euclidean distance between two points
    
    Args:
        point1: First point (x, y)
        point2: Second point (x, y)
        
    Returns:
        distance: Euclidean distance
    """
    if point1 is None or point2 is None:
        return None
    
    a = np.array(point1)
    b = np.array(point2)
    
    distance = np.linalg.norm(a - b)
    
    return distance

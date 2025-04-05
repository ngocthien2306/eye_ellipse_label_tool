import cv2
import numpy as np
import math
import os
from PyQt5.QtGui import QIcon, QPixmap, QPainter, QPen, QColor, QBrush
from PyQt5.QtCore import Qt, QSize

def create_checkmark_icon(size=16, color="#4CAF50"):
    """
    Creates a checkmark icon to indicate labeled images
    
    Args:
        size (int): Size of the icon in pixels
        color (str): Hex color code for the checkmark
    
    Returns:
        QIcon: A checkmark icon
    """
    # Create a transparent pixmap
    pixmap = QPixmap(size, size)
    pixmap.fill(Qt.transparent)
    
    # Set up painter
    painter = QPainter(pixmap)
    painter.setRenderHint(QPainter.Antialiasing)
    
    # Create pen for drawing
    pen = QPen()
    pen.setWidth(2)
    pen.setColor(QColor(color))
    painter.setPen(pen)
    
    # Draw checkmark
    painter.drawLine(3, size // 2, size // 2 - 1, size - 4)
    painter.drawLine(size // 2 - 1, size - 4, size - 3, 3)
    
    # End painting
    painter.end()
    
    # Create icon from pixmap
    return QIcon(pixmap)

def create_ellipse_icon(size=16, border_color="#3F51B5", fill_color="#E3F2FD"):
    """
    Creates an ellipse icon
    
    Args:
        size (int): Size of the icon in pixels
        border_color (str): Hex color code for the ellipse border
        fill_color (str): Hex color code for the ellipse fill
    
    Returns:
        QIcon: An ellipse icon
    """
    # Create a transparent pixmap
    pixmap = QPixmap(size, size)
    pixmap.fill(Qt.transparent)
    
    # Set up painter
    painter = QPainter(pixmap)
    painter.setRenderHint(QPainter.Antialiasing)
    
    # Create pen for drawing
    pen = QPen()
    pen.setWidth(1)
    pen.setColor(QColor(border_color))
    painter.setPen(pen)
    
    # Create brush for fill
    brush = QBrush(QColor(fill_color))
    painter.setBrush(brush)
    
    # Draw ellipse
    painter.drawEllipse(2, 4, size - 4, size - 8)
    
    # End painting
    painter.end()
    
    # Create icon from pixmap
    return QIcon(pixmap)

def create_warning_icon(size=16, color="#FFC107"):
    """
    Creates a warning triangle icon
    
    Args:
        size (int): Size of the icon in pixels
        color (str): Hex color code for the warning triangle
    
    Returns:
        QIcon: A warning triangle icon
    """
    # Create a transparent pixmap
    pixmap = QPixmap(size, size)
    pixmap.fill(Qt.transparent)
    
    # Set up painter
    painter = QPainter(pixmap)
    painter.setRenderHint(QPainter.Antialiasing)
    
    # Create pen for drawing
    pen = QPen()
    pen.setWidth(1)
    pen.setColor(QColor("#424242"))
    painter.setPen(pen)
    
    # Create brush for fill
    brush = QBrush(QColor(color))
    painter.setBrush(brush)
    
    # Draw warning triangle
    points = [
        (size // 2, 2),
        (size - 2, size - 2),
        (2, size - 2)
    ]
    painter.drawPolygon(*points)
    
    # Draw exclamation mark
    pen.setWidth(2)
    pen.setColor(QColor("#424242"))
    painter.setPen(pen)
    painter.setBrush(Qt.NoBrush)
    painter.drawLine(size // 2, 5, size // 2, size - 6)
    painter.drawPoint(size // 2, size - 4)
    
    # End painting
    painter.end()
    
    # Create icon from pixmap
    return QIcon(pixmap)

def draw_ellipse_on_image(image, ellipse_params, color=(0, 255, 0), thickness=2, center_color=(255, 0, 0)):
    """Draw an ellipse on the given image using the provided parameters"""
    if ellipse_params is None:
        return image
    
    result_image = image.copy()
    center, axes, angle = ellipse_params
    center = tuple(map(int, center))
    axes = tuple(map(int, axes))
    
    # Draw the ellipse
    cv2.ellipse(result_image, ellipse_params, color, thickness)
    
    # Draw center point
    cv2.circle(result_image, center, 5, center_color, -1)
    
    return result_image

def normalize_ellipse_params(ellipse_params, img_width, img_height):
    """Convert ellipse parameters to normalized YOLO format"""
    if ellipse_params is None:
        return None
    
    center, axes, angle = ellipse_params
    
    # Normalize parameters
    center_x_norm = center[0] / img_width
    center_y_norm = center[1] / img_height
    axes_x_norm = axes[0] / img_width
    axes_y_norm = axes[1] / img_height
    
    return (center_x_norm, center_y_norm), (axes_x_norm, axes_y_norm), angle

def denormalize_ellipse_params(ellipse_data, img_width, img_height):
    """Convert normalized YOLO format ellipse to pixel coordinates"""
    parts = ellipse_data.split()
    if len(parts) >= 6:
        class_id = int(parts[0])
        
        center_x = float(parts[1]) * img_width
        center_y = float(parts[2]) * img_height
        axes_x = float(parts[3]) * img_width
        axes_y = float(parts[4]) * img_height
        angle = float(parts[5])
        
        return ((center_x, center_y), (axes_x, axes_y), angle), class_id
    
    return None, None

def draw_segments_on_image(image, segment_data, point_color=(0, 0, 255), line_color=(0, 0, 255), 
                          text_color=(0, 255, 255), point_size=5, line_thickness=2):
    """Draw segmentation points and lines on the given image"""
    parts = segment_data.split()
    if len(parts) > 1:  # Make sure we have points
        # Class ID and points
        class_id = int(parts[0])
        points_data = parts[1:]
        
        img_height, img_width = image.shape[:2]
        
        # Convert normalized coordinates to pixel coordinates
        points = []
        for i in range(0, len(points_data), 2):
            if i + 1 < len(points_data):
                x = float(points_data[i]) * img_width
                y = float(points_data[i + 1]) * img_height
                points.append((int(x), int(y)))
        
        # Draw the polygon
        if len(points) > 2:
            points_array = np.array(points, dtype=np.int32)
            cv2.polylines(image, [points_array], True, line_color, line_thickness)
            
            # Number the points
            for i, point in enumerate(points):
                cv2.circle(image, point, point_size, point_color, -1)
                cv2.putText(image, str(i+1), 
                            (point[0]+5, point[1]-5), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color, 1)
        
        return points
    
    return []

def normalize_points(points, img_width, img_height):
    """Convert points to normalized YOLO format"""
    normalized_points = []
    for point in points:
        x_norm = point[0] / img_width
        y_norm = point[1] / img_height
        normalized_points.append(x_norm)
        normalized_points.append(y_norm)
    
    return normalized_points

def fit_ellipse_to_points(points):
    """Fit an ellipse to the given points"""
    if len(points) < 5:
        return None
    
    try:
        points_array = np.array(points, dtype=np.int32)
        ellipse_params = cv2.fitEllipse(points_array)
        return ellipse_params
    except Exception as e:
        print(f"Error fitting ellipse: {str(e)}")
        return None
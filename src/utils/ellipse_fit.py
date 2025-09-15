import cv2
import numpy as np
from skimage.measure import EllipseModel


def fit_ellipse_standard(points):
    if len(points) >= 5:
        center, axes, angle = cv2.fitEllipse(points)
        return {"center": center, "axes": axes, "angle": angle}
    return None

def fit_ellipse_direct(points):
    if len(points) >= 5:
        points_float = points.reshape(-1, 1, 2).astype(np.float32)
        try:
            center, axes, angle = cv2.fitEllipseDirect(points_float)
            return {"center": center, "axes": axes, "angle": angle}
        except:
            center, axes, angle = cv2.fitEllipse(points)
            return {"center": center, "axes": axes, "angle": angle}
    return None

def fit_ellipse_skimage(points):
    try:
        # Convert points to the format expected by EllipseModel
        points_array = points.reshape(-1, 2).astype(np.float64)
        
        # Fit ellipse using EllipseModel
        ellipse_model = EllipseModel()
        if not ellipse_model.estimate(points_array):
            # If estimation fails, fall back to OpenCV method
            center, axes, angle = cv2.fitEllipse(points)
            return {"center": center, "axes": axes, "angle": angle}
        
        # Extract parameters
        xc, yc, a, b, theta = ellipse_model.params
        
        # Convert parameters to dictionary format
        center = (xc, yc)
        axes = (2*a, 2*b)  # Full axes lengths (not semi-axes)
        angle_deg = np.degrees(theta) % 180  # Convert to degrees
        
        return {"center": center, "axes": axes, "angle": angle_deg}
    except:
        # Fall back to standard method if any error occurs
        center, axes, angle = cv2.fitEllipse(points)
        return {"center": center, "axes": axes, "angle": angle}

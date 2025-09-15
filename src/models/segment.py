import time
import cv2
import numpy as np
import onnxruntime

def resize_and_pad(image, target_size):
    """Resize and pad image to target size while maintaining aspect ratio"""
    target_width, target_height = target_size
    orig_height, orig_width = image.shape[:2]  # Handle both grayscale and RGB images
    
    # Calculate scaling factor
    scale = min(target_width / orig_width, target_height / orig_height)
    
    # Calculate new dimensions
    new_width = int(orig_width * scale)
    new_height = int(orig_height * scale)
    
    # Resize image
    resized_image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_LINEAR)
    
    # Create padded image of target size
    if len(image.shape) == 3:  # RGB image
        padded_image = np.zeros((target_height, target_width, image.shape[2]), dtype=np.uint8)
    else:  # Grayscale image
        padded_image = np.zeros((target_height, target_width), dtype=np.uint8)
    
    # Calculate padding
    pad_x = (target_width - new_width) // 2
    pad_y = (target_height - new_height) // 2
    
    # Place resized image onto padded image
    if len(image.shape) == 3:  # RGB image
        padded_image[pad_y:pad_y+new_height, pad_x:pad_x+new_width, :] = resized_image
    else:  # Grayscale image
        padded_image[pad_y:pad_y+new_height, pad_x:pad_x+new_width] = resized_image
    
    return padded_image, scale, pad_x, pad_y, new_width, new_height

def preprocess_image(image, target_size=(400, 400)):
    """Preprocess image for inference"""
    # Convert to RGB for visualization
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    original_image = image_rgb.copy()
    orig_height, orig_width = image_rgb.shape[:2]
    
    # Convert to grayscale for processing
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Resize and pad
    padded_image, scale, pad_x, pad_y, new_width, new_height = resize_and_pad(gray_image, target_size)
    
    # Convert padded grayscale to RGB for visualization
    padded_image_rgb = cv2.cvtColor(padded_image, cv2.COLOR_GRAY2RGB)
    
    # Normalize to [0,1] and prepare as tensor for ONNX
    image_tensor = padded_image.astype(np.float32) / 255.0
    
    # Add batch and channel dimensions [1, 1, H, W]
    image_tensor = np.expand_dims(np.expand_dims(image_tensor, axis=0), axis=0)
    
    # Store transformation parameters
    transform_params = {
        'scale': scale,
        'pad_x': pad_x,
        'pad_y': pad_y,
        'new_width': new_width,
        'new_height': new_height,
        'orig_width': orig_width,
        'orig_height': orig_height
    }
    
    return original_image, padded_image_rgb, image_tensor, transform_params

def run_inference(ort_session, image_tensor, verbose=True):
    """Run inference using ONNX runtime"""
    start_time = time.time()
    
    # Prepare input for ONNX runtime
    ort_inputs = {"input": image_tensor}
    
    # Run inference
    ort_outputs = ort_session.run(None, ort_inputs)
    output = ort_outputs[0]
    
    # Process output based on shape
    if output.shape[1] == 1:
        # Binary segmentation
        pred_mask = (output > 0.5).astype(np.uint8)
        pred_mask = pred_mask[0, 0]
    else:
        # Multi-class segmentation
        pred_mask = np.argmax(output, axis=1)
        pred_mask = pred_mask[0]
    
    end_time = time.time()
    if verbose:
        print("Inference Time: " + str(round((end_time-start_time) * 1000, 2)) + " ms")
    
    return pred_mask

def fit_ellipse_to_mask(mask):
    """Fit an ellipse to the largest contour in the mask"""
    # Find contours in the mask
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    
    if not contours:
        return None, False
    
    # Find the largest contour
    largest_contour = max(contours, key=cv2.contourArea)
    
    # Need at least 5 points to fit an ellipse
    if len(largest_contour) < 5:
        return None, False
    
    try:
        ellipse = cv2.fitEllipse(largest_contour)
        return ellipse, True
    except Exception as e:
        print(f"Error fitting ellipse: {e}")
        return None, False

def transform_ellipse_to_original(ellipse, transform_params):
    """Transform ellipse coordinates from padded space to original image space"""
    if ellipse is None:
        return None
    
    scale = transform_params['scale']
    pad_x = transform_params['pad_x']
    pad_y = transform_params['pad_y']
    
    (center_x, center_y), (axes_width, axes_height), angle = ellipse
    
    # Transform center coordinates
    orig_center_x = (center_x - pad_x) / scale
    orig_center_y = (center_y - pad_y) / scale
    
    # Transform axes
    orig_axes_width = axes_width / scale
    orig_axes_height = axes_height / scale
    
    return ((orig_center_x, orig_center_y), (orig_axes_width, orig_axes_height), angle)

def visualize_on_original(original_image, ellipse, transform_params=None):
    """Draw the ellipse on the original image"""
    overlay = original_image.copy()
    
    # Transform ellipse to original image space if needed
    if transform_params is not None and ellipse is not None:
        ellipse = transform_ellipse_to_original(ellipse, transform_params)
    
    # Draw ellipse if available
    if ellipse is not None:
        ellipse_color = (255, 0, 0)  # Red
        center, axes, angle = ellipse
        center = (int(center[0]), int(center[1]))
        axes = (int(axes[0]/2), int(axes[1]/2))
        
        cv2.ellipse(overlay, center, axes, angle, 0, 360, ellipse_color, 2)
        
        # Draw center point
        cv2.circle(overlay, center, 3, (255, 255, 0), -1)  # Yellow
    
    return overlay

def main():
    # Load ONNX model
    onnx_model_path = "mobilenet_v2_DeepLabV3Plus.onnx"
    print(f"Loading ONNX model: {onnx_model_path}")
    ort_session = onnxruntime.InferenceSession(
        onnx_model_path, providers=["CPUExecutionProvider"]
    )

    # Load and process image
    image_path = r"test\images\label_163_63_1067_720_bao_right_smooth_02_04_2025_part3_1189588055_353_1060.png"
    target_size = (224, 224)
    image = cv2.imread(str(image_path))
    
    if image is None:
        print(f"Error: Could not read image from {image_path}")
        return
    
    # Preprocess image
    start_time = time.time()
    original_image, padded_image, image_tensor, transform_params = preprocess_image(image, target_size)
    end_time = time.time()
    print("Pre-processing Time: " + str(round((end_time-start_time) * 1000, 2)) + " ms")
    
    # Run inference
    pred_mask = run_inference(ort_session, image_tensor)
    
    # Fit ellipse to mask
    start_time = time.time()
    ellipse, ellipse_success = fit_ellipse_to_mask(pred_mask)
    end_time = time.time()
    print("Post-processing Time: " + str(round((end_time-start_time) * 1000, 2)) + " ms")
    
    # Save mask
    cv2.imwrite("original_mask.png", pred_mask * 255)
    
    # Visualize and save results
    padded_result = visualize_on_original(padded_image, ellipse)
    cv2.imwrite("padded_result.png", cv2.cvtColor(padded_result, cv2.COLOR_RGB2BGR))
    
    original_result = visualize_on_original(original_image, ellipse, transform_params)
    cv2.imwrite("original_result.png", cv2.cvtColor(original_result, cv2.COLOR_RGB2BGR))
    
    # Print ellipse information
    if ellipse_success:
        print("Ellipse in padded space:", ellipse)
        original_ellipse = transform_ellipse_to_original(ellipse, transform_params)
        print("Ellipse in original space:", original_ellipse)
    else:
        print("Failed to fit ellipse to the mask")

if __name__ == "__main__":
    main()
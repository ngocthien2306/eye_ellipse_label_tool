import os
import cv2
import numpy as np
from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QGroupBox, QGridLayout,
                             QListWidget, QLabel, QPushButton, QFileDialog, QMessageBox, 
                             QScrollArea, QSlider, QTextEdit, QSplitter, QCheckBox)
from PyQt5.QtGui import QPixmap, QImage, QPainter, QPen, QColor, QCursor
from PyQt5.QtCore import Qt, QRect, QPoint

from styles import AnnotationStyles

class CropImageLabel(QLabel):
    """Custom QLabel that supports crop rectangle drawing and manipulation"""
    def __init__(self, *args, **kwargs):
        super(CropImageLabel, self).__init__(*args, **kwargs)
        self.setMouseTracking(True)  # Enable mouse tracking for hover effects

class SplitTab(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.parent = parent
        
        self.current_split_image_path = None
        self.current_split_image = None
        self.current_split_segment = None
        self.current_split_ellipse = None
        
        # Crop rectangle parameters
        self.crop_rect = None  # Format: (x, y, width, height)
        self.is_drawing_rect = False
        self.is_moving_rect = False
        self.is_resizing_rect = False
        self.rect_start_pos = None
        self.rect_resize_corner = None
        self.drag_start_pos = None
        
        self.setup_ui()
        
    def setup_ui(self):
        split_layout = QVBoxLayout()
        
        # Top section with load button
        top_layout = QHBoxLayout()
        self.split_folder_button = QPushButton("Load Data Folder")
        self.split_folder_button.clicked.connect(self.select_split_folder)
        top_layout.addWidget(self.split_folder_button)
        
        # Crop option
        self.crop_checkbox = QCheckBox("Enable Crop for Eye Region")
        self.crop_checkbox.setChecked(False)
        self.crop_checkbox.stateChanged.connect(self.crop_option_changed)
        top_layout.addWidget(self.crop_checkbox)
        
        # Split ratio controls
        ratio_group = QGroupBox("Split Ratios")
        ratio_layout = QGridLayout()
        
        self.train_ratio = QSlider(Qt.Horizontal)
        self.train_ratio.setRange(0, 100)
        self.train_ratio.setValue(70)
        self.train_ratio.valueChanged.connect(self.update_split_ratios)
        
        self.val_ratio = QSlider(Qt.Horizontal)
        self.val_ratio.setRange(0, 100)
        self.val_ratio.setValue(15)
        self.val_ratio.valueChanged.connect(self.update_split_ratios)
        
        self.test_ratio = QSlider(Qt.Horizontal)
        self.test_ratio.setRange(0, 100)
        self.test_ratio.setValue(15)
        self.test_ratio.setEnabled(False)  # This will be calculated
        
        self.train_label = QLabel("Train: 70%")
        self.val_label = QLabel("Validation: 15%")
        self.test_label = QLabel("Test: 15%")
        
        ratio_layout.addWidget(QLabel("Train:"), 0, 0)
        ratio_layout.addWidget(self.train_ratio, 0, 1)
        ratio_layout.addWidget(self.train_label, 0, 2)
        
        ratio_layout.addWidget(QLabel("Validation:"), 1, 0)
        ratio_layout.addWidget(self.val_ratio, 1, 1)
        ratio_layout.addWidget(self.val_label, 1, 2)
        
        ratio_layout.addWidget(QLabel("Test:"), 2, 0)
        ratio_layout.addWidget(self.test_ratio, 2, 1)
        ratio_layout.addWidget(self.test_label, 2, 2)
        
        ratio_group.setLayout(ratio_layout)
        top_layout.addWidget(ratio_group)
        
        self.split_button = QPushButton("Split Data")
        self.split_button.clicked.connect(self.split_data)
        self.split_button.setEnabled(False)
        self.split_button.setStyleSheet(AnnotationStyles.OUTLINE_BUTTON_STYLE)
        top_layout.addWidget(self.split_button)
        
        # Main content area with splitter
        split_content = QSplitter(Qt.Horizontal)
        
        # Left panel - file list
        left_widget = QWidget()
        left_layout = QVBoxLayout()
        
        self.split_file_list = QListWidget()
        self.split_file_list.itemClicked.connect(self.load_split_image)
        
        left_layout.addWidget(QLabel("Image Files:"))
        left_layout.addWidget(self.split_file_list)
        left_widget.setLayout(left_layout)
        
        # Right panel - image and labels view
        right_widget = QWidget()
        right_layout = QVBoxLayout()
        
        # Create scroll area for the image
        self.split_scroll_area = QScrollArea()
        self.split_scroll_area.setWidgetResizable(True)
        self.split_scroll_area.setMinimumSize(650, 550)
        
        # Create custom image label inside scroll area
        self.split_image_label = CropImageLabel("Select an image to preview")
        self.split_image_label.setAlignment(Qt.AlignCenter)
        self.split_image_label.mousePressEvent = self.on_mouse_press
        self.split_image_label.mouseMoveEvent = self.on_mouse_move
        self.split_image_label.mouseReleaseEvent = self.on_mouse_release
        self.split_scroll_area.setWidget(self.split_image_label)
        
        # Zoom controls
        split_zoom_layout = QHBoxLayout()
        
        self.split_zoom_out_button = QPushButton("-")
        self.split_zoom_out_button.clicked.connect(self.split_zoom_out)
        self.split_zoom_out_button.setFixedWidth(40)
        
        self.split_zoom_slider = QSlider(Qt.Horizontal)
        self.split_zoom_slider.setMinimum(int(self.parent.min_zoom * 100))
        self.split_zoom_slider.setMaximum(int(self.parent.max_zoom * 100))
        self.split_zoom_slider.setValue(int(self.parent.zoom_factor * 100))
        self.split_zoom_slider.valueChanged.connect(self.split_set_zoom)
        
        self.split_zoom_in_button = QPushButton("+")
        self.split_zoom_in_button.clicked.connect(self.split_zoom_in)
        self.split_zoom_in_button.setFixedWidth(40)
        
        self.split_zoom_reset_button = QPushButton("Reset Zoom")
        self.split_zoom_reset_button.clicked.connect(self.split_reset_zoom)
        
        self.split_zoom_label = QLabel("100%")
        self.split_zoom_label.setFixedWidth(60)
        
        split_zoom_layout.addWidget(self.split_zoom_out_button)
        split_zoom_layout.addWidget(self.split_zoom_slider)
        split_zoom_layout.addWidget(self.split_zoom_in_button)
        split_zoom_layout.addWidget(self.split_zoom_label)
        split_zoom_layout.addWidget(self.split_zoom_reset_button)
        
        # Label info area
        self.split_info_text = QTextEdit()
        self.split_info_text.setReadOnly(True)
        self.split_info_text.setMaximumHeight(100)
        
        right_layout.addWidget(self.split_scroll_area)
        right_layout.addLayout(split_zoom_layout)
        right_layout.addWidget(QLabel("Label Information:"))
        right_layout.addWidget(self.split_info_text)
        right_widget.setLayout(right_layout)
        
        # Add to splitter
        split_content.addWidget(left_widget)
        split_content.addWidget(right_widget)
        split_content.setSizes([300, 700])
        
        # Enable mouse wheel for zooming
        self.split_scroll_area.wheelEvent = self.split_wheel_event
        
        # Add to main layout
        split_layout.addLayout(top_layout)
        split_layout.addWidget(split_content)
        
        self.setLayout(split_layout)
        
    def select_split_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "Select Data Folder")
        if folder:
            self.split_folder = folder
            self.images_dir = os.path.join(folder, "images")
            self.labels_dir = os.path.join(folder, "labels")
            self.labels_ellipse_dir = os.path.join(folder, "labels_ellipse")
            
            # Check if required folders exist
            if not os.path.exists(self.images_dir) or not os.path.exists(self.labels_dir):
                QMessageBox.warning(self, "Warning", 
                    "The selected folder does not contain the required structure:\n"
                    "- images\n- labels\n- labels_ellipse")
                return
                
            self.load_split_file_list()
            self.split_button.setEnabled(True)
            self.update_info_text(f"Loaded split data folder: {folder}")

    def load_split_file_list(self):
        self.split_file_list.clear()
        if not hasattr(self, 'images_dir') or not os.path.exists(self.images_dir):
            return
            
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif']
        self.split_image_files = []
        
        for file in os.listdir(self.images_dir):
            file_path = os.path.join(self.images_dir, file)
            if os.path.isfile(file_path):
                ext = os.path.splitext(file)[1].lower()
                if ext in image_extensions:
                    self.split_file_list.addItem(file)
                    self.split_image_files.append(file)
                    
        self.update_info_text(f"Found {len(self.split_image_files)} images in the dataset")
    
    def load_split_image(self, item):
        if not item:
            return
            
        file_name = item.text()
        self.current_split_image_path = os.path.join(self.images_dir, file_name)
        self.current_split_image = cv2.imread(self.current_split_image_path)
        
        if self.current_split_image is not None:
            base_name = os.path.splitext(file_name)[0]
            
            # Look for corresponding label files
            segment_path = os.path.join(self.labels_dir, f"{base_name}.txt")
            ellipse_path = os.path.join(self.labels_ellipse_dir, f"{base_name}.txt")
            
            has_segment = os.path.exists(segment_path)
            has_ellipse = os.path.exists(ellipse_path)
            
            # Store label data
            self.current_split_segment = None
            self.current_split_ellipse = None
            
            if has_segment:
                with open(segment_path, 'r') as f:
                    self.current_split_segment = f.read().strip()
                    
            if has_ellipse:
                with open(ellipse_path, 'r') as f:
                    self.current_split_ellipse = f.read().strip()
            
            # Display the image with labels
            image = self.display_split_image()
            
            # Display label info
            info = f"Image: {file_name}\n"
            if has_segment:
                info += f"Has segmentation label: Yes\n"
            else:
                info += f"Has segmentation label: No\n"
                
            if has_ellipse:
                info += f"Has ellipse label: Yes\n"
                parts = self.current_split_ellipse.split()
                if len(parts) >= 6:
                    # Parse ellipse parameters: class_id, center_x, center_y, axes_x, axes_y, angle
                    class_id = int(parts[0])
                    
                    img_height, img_width = image.shape[:2]
                    
                    center_x = float(parts[1]) * img_width
                    center_y = float(parts[2]) * img_height
                    axes_x = float(parts[3]) * img_width
                    axes_y = float(parts[4]) * img_height
                    angle = float(parts[5])
                    
                    # Draw the ellipse
                    center = (int(center_x), int(center_y))
                    axes = (int(axes_x), int(axes_y))
                    info += f"Ellipse center: {center}\n"
                    info += f"Axes: {axes}\n"
                    info += f"Angle: {round(angle,2)} degrees\n"
            else:
                info += f"Has ellipse label: No"
                
            self.split_info_text.setPlainText(info)
            
    def display_split_image(self):
        if self.current_split_image is None:
            return
        
        display_image = self.current_split_image.copy()
        
        # Draw segment if available
        if self.current_split_segment:
            self.draw_segment_on_image(display_image, self.current_split_segment)
        
        # Draw ellipse if available
        if self.current_split_ellipse:
            self.draw_ellipse_on_image(display_image, self.current_split_ellipse)
        
        # Convert to RGB for Qt
        display_image = cv2.cvtColor(display_image, cv2.COLOR_BGR2RGB)
        
        height, width, channel = display_image.shape
        bytes_per_line = 3 * width
        q_image = QImage(display_image.data, width, height, bytes_per_line, QImage.Format_RGB888)
        
        pixmap = QPixmap.fromImage(q_image)
        
        # Draw crop rectangle if enabled
        if self.crop_checkbox.isChecked() and self.crop_rect:
            rect_x, rect_y, rect_w, rect_h = self.crop_rect
            
            # Create a painter to draw on the pixmap
            painter = QPainter(pixmap)
            pen = QPen(QColor(255, 0, 255))  # Magenta
            pen.setWidth(2)
            pen.setStyle(Qt.DashLine)
            painter.setPen(pen)
            painter.drawRect(rect_x, rect_y, rect_w, rect_h)
            
            # Draw corner handles
            handle_size = 6
            painter.setBrush(QColor(255, 0, 255))
            painter.drawRect(rect_x - handle_size//2, rect_y - handle_size//2, handle_size, handle_size)  # Top-left
            painter.drawRect(rect_x + rect_w - handle_size//2, rect_y - handle_size//2, handle_size, handle_size)  # Top-right
            painter.drawRect(rect_x - handle_size//2, rect_y + rect_h - handle_size//2, handle_size, handle_size)  # Bottom-left
            painter.drawRect(rect_x + rect_w - handle_size//2, rect_y + rect_h - handle_size//2, handle_size, handle_size)  # Bottom-right
            painter.end()
        
        # Apply zoom factor
        scaled_width = int(width * self.parent.zoom_factor)
        scaled_height = int(height * self.parent.zoom_factor)
        
        pixmap = pixmap.scaled(scaled_width, scaled_height, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        
        self.split_image_label.setPixmap(pixmap)
        self.split_image_label.setFixedSize(pixmap.size())
        return display_image
    
    def draw_segment_on_image(self, image, segment_data):
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
                cv2.polylines(image, [points_array], True, (0, 0, 255), 2)
                
                # Number the points
                for i, point in enumerate(points):
                    cv2.circle(image, point, 5, (0, 0, 255), -1)
                    cv2.putText(image, str(i+1), 
                                (point[0]+5, point[1]-5), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

    def draw_ellipse_on_image(self, image, ellipse_data):
        parts = ellipse_data.split()
        if len(parts) >= 6:
            # Parse ellipse parameters: class_id, center_x, center_y, axes_x, axes_y, angle
            class_id = int(parts[0])
            
            img_height, img_width = image.shape[:2]
            
            center_x = float(parts[1]) * img_width
            center_y = float(parts[2]) * img_height
            axes_x = float(parts[3]) * img_width
            axes_y = float(parts[4]) * img_height
            angle = float(parts[5])
            
            # Draw the ellipse
            center = (int(center_x), int(center_y))
            axes = (int(axes_x), int(axes_y))
            
            cv2.ellipse(image, (center, axes, angle), (0, 255, 0), 2)
            
            # Draw center point
            cv2.circle(image, center, 5, (255, 0, 0), -1)
            
    def split_wheel_event(self, event):
        delta = event.angleDelta().y()
        if delta > 0:
            self.split_zoom_in()
        else:
            self.split_zoom_out()
        event.accept()

    def split_zoom_in(self):
        if self.parent.zoom_factor < self.parent.max_zoom:
            self.parent.zoom_factor += 0.1
            self.update_split_zoom()

    def split_zoom_out(self):
        if self.parent.zoom_factor > self.parent.min_zoom:
            self.parent.zoom_factor -= 0.1
            self.update_split_zoom()

    def split_reset_zoom(self):
        self.parent.zoom_factor = 1.0
        self.update_split_zoom()

    def split_set_zoom(self, value):
        self.parent.zoom_factor = value / 100
        self.update_split_zoom(from_slider=True)

    def update_split_zoom(self, from_slider=False):
        try:
            if self.current_split_image is not None:
                self.parent.zoom_factor = max(self.parent.min_zoom, min(self.parent.max_zoom, self.parent.zoom_factor))
                
                if not from_slider:
                    self.split_zoom_slider.setValue(int(self.parent.zoom_factor * 100))
                
                self.split_zoom_label.setText(f"{int(self.parent.zoom_factor * 100)}%")
                self.display_split_image()
                
        except Exception as e:
            error_msg = f"Please select an image"
            QMessageBox.critical(self, "Error", error_msg)

    def update_split_ratios(self):
        train_val = self.train_ratio.value() + self.val_ratio.value()
        
        # Ensure total is 100%
        if train_val > 100:
            if self.sender() == self.train_ratio:
                self.val_ratio.setValue(100 - self.train_ratio.value())
            else:
                self.train_ratio.setValue(100 - self.val_ratio.value())
        
        # Update test ratio (always calculated)
        test = 100 - (self.train_ratio.value() + self.val_ratio.value())
        
        # Update labels
        self.train_label.setText(f"Train: {self.train_ratio.value()}%")
        self.val_label.setText(f"Validation: {self.val_ratio.value()}%")
        self.test_label.setText(f"Test: {test}%")

    def copy_files_to_subset(self, subset, files, output_dir, output_dir_cropped=None):
        """Copy image and label files to the appropriate subset directory, with optional cropped versions.
        
        Args:
            subset (str): Subset name ('train', 'valid', or 'test')
            files (list): List of image filenames to copy
            output_dir (str): Path to the main output directory
            output_dir_cropped (str, optional): Path to the cropped output directory
        """
        import shutil
        
        for file in files:
            base_name = os.path.splitext(file)[0]
            ext = os.path.splitext(file)[1]
            
            # Source paths
            img_src = os.path.join(self.images_dir, file)
            label_src = os.path.join(self.labels_dir, f"{base_name}.txt")
            ellipse_src = os.path.join(self.labels_ellipse_dir, f"{base_name}.txt")
            
            # Destination paths for normal version
            img_dst = os.path.join(output_dir, subset, 'images', file)
            label_dst = os.path.join(output_dir, subset, 'labels', f"{base_name}.txt")
            ellipse_dst = os.path.join(output_dir, subset, 'labels_ellipse', f"{base_name}.txt")
            
            # Copy files for normal version if they exist
            if os.path.exists(img_src):
                shutil.copy2(img_src, img_dst)
                
            if os.path.exists(label_src):
                shutil.copy2(label_src, label_dst)
                
            if os.path.exists(ellipse_src):
                shutil.copy2(ellipse_src, ellipse_dst)
            
            # Handle cropped version if enabled
            if output_dir_cropped and os.path.exists(img_src) and os.path.exists(ellipse_src):
                try:
                    # Load and process the image
                    img = cv2.imread(img_src)
                    if img is None:
                        continue  # Skip if image can't be loaded
                    
                    # Load the ellipse data for cropping reference
                    with open(ellipse_src, 'r') as f:
                        ellipse_data = f.read().strip()
                    
                    # Determine crop rectangle
                    rect = self.create_crop_rect_for_image(img, ellipse_data)
                    if not rect:
                        continue  # Skip if no valid crop rectangle
                    
                    # Extract crop coordinates
                    x, y, w, h = rect
                    
                    # Ensure crop dimensions are valid
                    if w <= 0 or h <= 0 or x >= img.shape[1] or y >= img.shape[0]:
                        continue
                    
                    # Perform the crop
                    cropped_img = img[y:y+h, x:x+w]
                    if cropped_img.size == 0:
                        continue  # Skip if resulting crop is empty
                    
                    # Prepare destination paths for cropped version
                    img_crop_dst = os.path.join(output_dir_cropped, subset, 'images', file)
                    label_crop_dst = os.path.join(output_dir_cropped, subset, 'labels', f"{base_name}.txt")
                    ellipse_crop_dst = os.path.join(output_dir_cropped, subset, 'labels_ellipse', f"{base_name}.txt")
                    
                    # Save cropped image
                    cv2.imwrite(img_crop_dst, cropped_img)
                    
                    # Adjust and save label files
                    if os.path.exists(label_src):
                        self.create_adjusted_label(
                            label_src, 
                            label_crop_dst,
                            img.shape, 
                            rect
                        )
                    
                    if os.path.exists(ellipse_src):
                        self.create_adjusted_ellipse(
                            ellipse_src,
                            ellipse_crop_dst,
                            img.shape,
                            rect
                        )
                except Exception as e:
                    # Log error but continue processing other files
                    print(f"Error processing cropped version of {file}: {str(e)}")
                    continue

    def split_data(self):
        if not hasattr(self, 'split_image_files') or not self.split_image_files:
            QMessageBox.warning(self, "Warning", "No images found to split!")
            return
        
        try:
            # Check if crop is enabled
            crop_enabled = self.crop_checkbox.isChecked()
            
            # Create output structure
            output_dir = os.path.join(self.split_folder, "split")
            os.makedirs(output_dir, exist_ok=True)
            
            # For cropped version if enabled
            if crop_enabled:
                output_dir_cropped = os.path.join(self.split_folder, "split_cropped")
                os.makedirs(output_dir_cropped, exist_ok=True)
            
            # Create train/val/test directories with subdirectories
            for subset in ['train', 'valid', 'test']:
                # Normal split directories
                subset_dir = os.path.join(output_dir, subset)
                os.makedirs(os.path.join(subset_dir, 'images'), exist_ok=True)
                os.makedirs(os.path.join(subset_dir, 'labels'), exist_ok=True)
                os.makedirs(os.path.join(subset_dir, 'labels_ellipse'), exist_ok=True)
                
                # Cropped split directories if enabled
                if crop_enabled:
                    subset_dir_cropped = os.path.join(output_dir_cropped, subset)
                    os.makedirs(os.path.join(subset_dir_cropped, 'images'), exist_ok=True)
                    os.makedirs(os.path.join(subset_dir_cropped, 'labels'), exist_ok=True)
                    os.makedirs(os.path.join(subset_dir_cropped, 'labels_ellipse'), exist_ok=True)
            
            # Get split ratios
            train_ratio = self.train_ratio.value() / 100
            val_ratio = self.val_ratio.value() / 100
            test_ratio = 1 - train_ratio - val_ratio
            
            # Shuffle file list
            import random
            random.shuffle(self.split_image_files)
            
            # Calculate sizes
            total_files = len(self.split_image_files)
            train_size = int(total_files * train_ratio)
            val_size = int(total_files * val_ratio)
            
            # Split into subsets
            train_files = self.split_image_files[:train_size]
            val_files = self.split_image_files[train_size:train_size+val_size]
            test_files = self.split_image_files[train_size+val_size:]
            
            # Copy files to respective directories
            self.copy_files_to_subset('train', train_files, output_dir, output_dir_cropped if crop_enabled else None)
            self.copy_files_to_subset('valid', val_files, output_dir, output_dir_cropped if crop_enabled else None)
            self.copy_files_to_subset('test', test_files, output_dir, output_dir_cropped if crop_enabled else None)
            
            # Show success message
            message = f"Data successfully split:\n" \
                    f"Train: {len(train_files)} files ({train_ratio*100:.1f}%)\n" \
                    f"Validation: {len(val_files)} files ({val_ratio*100:.1f}%)\n" \
                    f"Test: {len(test_files)} files ({test_ratio*100:.1f}%)\n\n" \
                    f"Split data saved to: {output_dir}"
                    
            if crop_enabled:
                message += f"\nCropped version saved to: {output_dir_cropped}"
                    
            self.update_info_text(message)
            QMessageBox.information(self, "Success", message)
            
        except Exception as e:
            error_msg = f"Error splitting data: {str(e)}"
            QMessageBox.critical(self, "Error", error_msg)
            self.update_info_text(error_msg)



    def crop_option_changed(self, state):
        """Handle crop checkbox state change"""
        if state == Qt.Checked:
            # Enable crop mode
            self.parent.status_bar.showMessage("Crop mode enabled. Draw a rectangle around the eye region.")
            if self.current_split_image is not None:
                # Initialize default crop rectangle if image is loaded
                h, w = self.current_split_image.shape[:2]
                self.crop_rect = [300, 200, 600, 400] # (int(w*0.25), int(h*0.25), int(w*0.5), int(h*0.5))
                self.display_split_image()
        else:
            # Disable crop mode
            self.crop_rect = None
            self.display_split_image()
            self.parent.status_bar.showMessage("Crop mode disabled.")
    
    def on_mouse_press(self, event):
        """Handle mouse press events for crop rectangle"""
        if not self.crop_checkbox.isChecked() or self.current_split_image is None:
            return
        
        pos = event.pos()
        # Convert position accounting for zoom
        x = int(pos.x() / self.parent.zoom_factor)
        y = int(pos.y() / self.parent.zoom_factor)
        pos_converted = QPoint(x, y)
        
        if self.crop_rect:
            # Check if clicking on existing rectangle
            rect_x, rect_y, rect_w, rect_h = self.crop_rect
            rect = QRect(rect_x, rect_y, rect_w, rect_h)
            
            # Check if clicking on a corner for resizing (with a small margin)
            margin = 10
            top_left = QRect(rect_x - margin, rect_y - margin, margin*2, margin*2)
            top_right = QRect(rect_x + rect_w - margin, rect_y - margin, margin*2, margin*2)
            bottom_left = QRect(rect_x - margin, rect_y + rect_h - margin, margin*2, margin*2)
            bottom_right = QRect(rect_x + rect_w - margin, rect_y + rect_h - margin, margin*2, margin*2)
            
            if top_left.contains(pos_converted):
                self.is_resizing_rect = True
                self.rect_resize_corner = "top_left"
                self.drag_start_pos = pos_converted
            elif top_right.contains(pos_converted):
                self.is_resizing_rect = True
                self.rect_resize_corner = "top_right"
                self.drag_start_pos = pos_converted
            elif bottom_left.contains(pos_converted):
                self.is_resizing_rect = True
                self.rect_resize_corner = "bottom_left"
                self.drag_start_pos = pos_converted
            elif bottom_right.contains(pos_converted):
                self.is_resizing_rect = True
                self.rect_resize_corner = "bottom_right"
                self.drag_start_pos = pos_converted
            elif rect.contains(pos_converted):
                # Moving the rectangle
                self.is_moving_rect = True
                self.drag_start_pos = pos_converted
            else:
                # Start drawing a new rectangle
                self.is_drawing_rect = True
                self.rect_start_pos = pos_converted
        else:
            # Start drawing a new rectangle
            self.is_drawing_rect = True
            self.rect_start_pos = pos_converted
    
    def on_mouse_move(self, event):
        """Handle mouse move events for crop rectangle"""
        if not self.crop_checkbox.isChecked() or self.current_split_image is None:
            return
        
        pos = event.pos()
        # Convert position accounting for zoom
        x = int(pos.x() / self.parent.zoom_factor)
        y = int(pos.y() / self.parent.zoom_factor)
        pos_converted = QPoint(x, y)
        
        h, w = self.current_split_image.shape[:2]
        
        if self.is_drawing_rect:
            # Drawing a new rectangle
            start_x = self.rect_start_pos.x()
            start_y = self.rect_start_pos.y()
            width = pos_converted.x() - start_x
            height = pos_converted.y() - start_y
            
            # Ensure rectangle stays within image bounds
            if start_x + width > w:
                width = w - start_x
            if start_y + height > h:
                height = h - start_y
                
            # Handle negative dimensions
            if width < 0:
                start_x = start_x + width
                width = abs(width)
            if height < 0:
                start_y = start_y + height
                height = abs(height)
            
            self.crop_rect = (start_x, start_y, width, height)
            self.display_split_image()
            
        elif self.is_moving_rect:
            # Moving the rectangle
            if self.crop_rect and self.drag_start_pos:
                rect_x, rect_y, rect_w, rect_h = self.crop_rect
                delta_x = pos_converted.x() - self.drag_start_pos.x()
                delta_y = pos_converted.y() - self.drag_start_pos.y()
                
                new_x = max(0, min(rect_x + delta_x, w - rect_w))
                new_y = max(0, min(rect_y + delta_y, h - rect_h))
                
                self.crop_rect = (new_x, new_y, rect_w, rect_h)
                self.drag_start_pos = pos_converted
                self.display_split_image()
                
        elif self.is_resizing_rect:
            # Resizing the rectangle
            if self.crop_rect and self.drag_start_pos and self.rect_resize_corner:
                rect_x, rect_y, rect_w, rect_h = self.crop_rect
                
                if self.rect_resize_corner == "top_left":
                    delta_x = pos_converted.x() - self.drag_start_pos.x()
                    delta_y = pos_converted.y() - self.drag_start_pos.y()
                    
                    new_x = max(0, rect_x + delta_x)
                    new_y = max(0, rect_y + delta_y)
                    new_w = max(10, rect_w - delta_x)
                    new_h = max(10, rect_h - delta_y)
                    
                elif self.rect_resize_corner == "top_right":
                    delta_x = pos_converted.x() - self.drag_start_pos.x()
                    delta_y = pos_converted.y() - self.drag_start_pos.y()
                    
                    new_x = rect_x
                    new_y = max(0, rect_y + delta_y)
                    new_w = max(10, min(rect_w + delta_x, w - rect_x))
                    new_h = max(10, rect_h - delta_y)
                    
                elif self.rect_resize_corner == "bottom_left":
                    delta_x = pos_converted.x() - self.drag_start_pos.x()
                    delta_y = pos_converted.y() - self.drag_start_pos.y()
                    
                    new_x = max(0, rect_x + delta_x)
                    new_y = rect_y
                    new_w = max(10, rect_w - delta_x)
                    new_h = max(10, min(rect_h + delta_y, h - rect_y))
                    
                elif self.rect_resize_corner == "bottom_right":
                    delta_x = pos_converted.x() - self.drag_start_pos.x()
                    delta_y = pos_converted.y() - self.drag_start_pos.y()
                    
                    new_x = rect_x
                    new_y = rect_y
                    new_w = max(10, min(rect_w + delta_x, w - rect_x))
                    new_h = max(10, min(rect_h + delta_y, h - rect_y))
                
                self.crop_rect = (new_x, new_y, new_w, new_h)
                self.drag_start_pos = pos_converted
                self.display_split_image()
        
        # Change cursor based on position
        if self.crop_rect:
            rect_x, rect_y, rect_w, rect_h = self.crop_rect
            rect = QRect(rect_x, rect_y, rect_w, rect_h)
            
            margin = 10
            top_left = QRect(rect_x - margin, rect_y - margin, margin*2, margin*2)
            top_right = QRect(rect_x + rect_w - margin, rect_y - margin, margin*2, margin*2)
            bottom_left = QRect(rect_x - margin, rect_y + rect_h - margin, margin*2, margin*2)
            bottom_right = QRect(rect_x + rect_w - margin, rect_y + rect_h - margin, margin*2, margin*2)
            
            if (top_left.contains(pos_converted) or bottom_right.contains(pos_converted)):
                self.split_image_label.setCursor(Qt.SizeFDiagCursor)
            elif (top_right.contains(pos_converted) or bottom_left.contains(pos_converted)):
                self.split_image_label.setCursor(Qt.SizeBDiagCursor)
            elif rect.contains(pos_converted):
                self.split_image_label.setCursor(Qt.SizeAllCursor)
            else:
                self.split_image_label.setCursor(Qt.ArrowCursor)
    
    def on_mouse_release(self, event):
        """Handle mouse release events for crop rectangle"""
        self.is_drawing_rect = False
        self.is_moving_rect = False
        self.is_resizing_rect = False
        self.rect_start_pos = None
        self.drag_start_pos = None
        
        if self.crop_rect and self.current_split_ellipse:
            # Verify if the crop rectangle contains the ellipse
            ellipse_center, ellipse_axes = self.get_ellipse_parameters()
            rect_x, rect_y, rect_w, rect_h = self.crop_rect
            
            if not (rect_x <= ellipse_center[0] <= rect_x + rect_w and 
                    rect_y <= ellipse_center[1] <= rect_y + rect_h):
                self.update_info_text("Warning: Crop rectangle must contain the ellipse center!", append=True)
    
    def get_ellipse_parameters(self):
        """Extract ellipse parameters from YOLO format"""
        if not self.current_split_ellipse:
            return None, None
        
        parts = self.current_split_ellipse.split()
        if len(parts) >= 6:
            img_height, img_width = self.current_split_image.shape[:2]
            
            center_x = float(parts[1]) * img_width
            center_y = float(parts[2]) * img_height
            axes_x = float(parts[3]) * img_width
            axes_y = float(parts[4]) * img_height
            
            return (int(center_x), int(center_y)), (int(axes_x), int(axes_y))
        
        return None, None
    
    def create_crop_rect_for_image(self, image, ellipse_data):
        """Create a crop rectangle for the given image based on ellipse or current crop rect"""
        h, w = image.shape[:2]
        
        # If we have a crop_rect already defined (from UI), use it
        if hasattr(self, 'crop_rect') and self.crop_rect:
            return self.crop_rect
        
        # Otherwise, create one based on the ellipse
        parts = ellipse_data.split()
        if len(parts) >= 6:
            # Parse ellipse parameters
            class_id = int(parts[0])
            center_x = float(parts[1]) * w
            center_y = float(parts[2]) * h
            axes_x = float(parts[3]) * w
            axes_y = float(parts[4]) * h
            
            # Create rectangle around ellipse with some margin
            margin = 1.5  # 50% margin around the ellipse
            rect_w = int(axes_x * 2 * margin)
            rect_h = int(axes_y * 2 * margin)
            rect_x = max(0, int(center_x - rect_w / 2))
            rect_y = max(0, int(center_y - rect_h / 2))
            
            # Ensure rectangle stays within image bounds
            if rect_x + rect_w > w:
                rect_w = w - rect_x
            if rect_y + rect_h > h:
                rect_h = h - rect_y
                
            return (rect_x, rect_y, rect_w, rect_h)
            
        return None
    
    def create_adjusted_label(self, src_path, dst_path, img_shape, crop_rect):
        """Create adjusted segmentation label for cropped image"""
        if not os.path.exists(src_path):
            return
            
        with open(src_path, 'r') as f:
            segment_data = f.read().strip()
            
        parts = segment_data.split()
        if len(parts) > 1:
            class_id = parts[0]
            points_data = parts[1:]
            
            img_height, img_width = img_shape[:2]
            crop_x, crop_y, crop_w, crop_h = crop_rect
            
            # Adjust points to new coordinate system
            new_points = []
            for i in range(0, len(points_data), 2):
                if i + 1 < len(points_data):
                    # Denormalize from original image
                    x = float(points_data[i]) * img_width
                    y = float(points_data[i + 1]) * img_height
                    
                    # Adjust to crop coordinates
                    x = x - crop_x
                    y = y - crop_y
                    
                    # Normalize to cropped dimensions
                    x_norm = x / crop_w
                    y_norm = y / crop_h
                    
                    # Ensure coordinates are within [0,1] range
                    x_norm = max(0, min(1, x_norm))
                    y_norm = max(0, min(1, y_norm))
                    
                    new_points.append(f"{x_norm:.15f}")
                    new_points.append(f"{y_norm:.15f}")
            
            # Create new label
            new_label = f"{class_id} " + " ".join(new_points)
            
            # Write adjusted label
            with open(dst_path, 'w') as f:
                f.write(new_label)
    
    def create_adjusted_ellipse(self, src_path, dst_path, img_shape, crop_rect):
        """Create adjusted ellipse label for cropped image"""
        if not os.path.exists(src_path):
            return
            
        with open(src_path, 'r') as f:
            ellipse_data = f.read().strip()
            
        parts = ellipse_data.split()
        if len(parts) >= 6:
            class_id = parts[0]
            
            img_height, img_width = img_shape[:2]
            crop_x, crop_y, crop_w, crop_h = crop_rect
            
            # Denormalize from original image
            center_x = float(parts[1]) * img_width
            center_y = float(parts[2]) * img_height
            axes_x = float(parts[3]) * img_width
            axes_y = float(parts[4]) * img_height
            angle = float(parts[5])
            
            # Adjust to crop coordinates
            center_x = center_x - crop_x
            center_y = center_y - crop_y
            
            # Normalize to cropped dimensions
            center_x_norm = center_x / crop_w
            center_y_norm = center_y / crop_h
            axes_x_norm = axes_x / crop_w
            axes_y_norm = axes_y / crop_h
            
            # Ensure coordinates are within [0,1] range
            center_x_norm = max(0, min(1, center_x_norm))
            center_y_norm = max(0, min(1, center_y_norm))
            
            # Create new ellipse label
            new_ellipse = f"{class_id} {center_x_norm:.15f} {center_y_norm:.15f} {axes_x_norm:.15f} {axes_y_norm:.15f} {angle:.15f}"
            
            # Write adjusted label
            with open(dst_path, 'w') as f:
                f.write(new_ellipse)
    
    def update_info_text(self, text, append=False):
        if append:
            current_text = self.split_info_text.toPlainText()
            self.split_info_text.setPlainText(f"{current_text}\n{text}")
        else:
            self.split_info_text.setPlainText(text)
        self.parent.status_bar.showMessage(text.split('\n')[0])
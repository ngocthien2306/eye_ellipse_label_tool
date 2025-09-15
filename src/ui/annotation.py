import os
import cv2
import numpy as np
from ultralytics import YOLO

from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QListWidget, QLabel, QPushButton, 
                             QFileDialog, QMessageBox, QScrollArea, QSlider, QListWidgetItem, 
                             QTextEdit, QSplitter, QCheckBox, QComboBox)
from PyQt5.QtGui import QPixmap, QImage, QIcon, QPainter, QColor, QPen
from PyQt5.QtCore import Qt, QPoint

from src.utils.styles import AnnotationStyles
from src.utils.utils import create_checkmark_icon
from src.utils.ellipse_fit import fit_ellipse_direct, fit_ellipse_skimage, fit_ellipse_standard
from src.utils.remove_noise import clean_contour_points

class AnnotationTab(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.parent = parent
        
        self.current_image_path = None
        self.current_image = None
        
        # Multi-object support
        self.current_class = 0  # 0: iris, 1: pupil
        self.class_names = ['iris', 'pupil']
        self.objects = {0: {'points': [], 'ellipse': None}, 1: {'points': [], 'ellipse': None}}
        self.deleted_points = {0: [], 1: []}  # Store deleted points for redo per class
        
        # Legacy support (will be removed)
        self.points = []
        self.ellipse_params = None
        
        self.flag_preview = True
        self.flag_ellipse = True
        self.model = YOLO("assets/models/best_seg.pt")

        self.use_model = False
        self.current_method = 2
        self.current_clean_method = 0
        self.crop_rect = [0, 0, 320, 320]
        
        self.segment_type = "instance" # semantic
        
        self.setup_ui()
        self.apply_styles()
    
    def setup_ui(self):
        # UI setup code remains the same as in your original file
        annotation_layout = QVBoxLayout()
        
        # Main content with file list, image view and info panel
        content_splitter = QSplitter(Qt.Horizontal)
        
        # Left panel - File list
        left_widget = QWidget()
        left_layout = QVBoxLayout()
        
        # Create folder selection header with label
        folder_header = QLabel("Image Folder")
        
        self.folder_button = QPushButton("Select Image Folder")
        self.folder_button.clicked.connect(self.select_folder)
        
        self.file_list = QListWidget()
        self.file_list.itemClicked.connect(self.load_image)
        
        # Navigation buttons
        nav_layout = QHBoxLayout()
        self.prev_file_button = QPushButton("◀ Prev")
        self.prev_file_button.clicked.connect(self.prev_file)
        self.next_file_button = QPushButton("Next ▶")
        self.next_file_button.clicked.connect(self.next_file)
        nav_layout.addWidget(self.prev_file_button)
        nav_layout.addWidget(self.next_file_button)
        
        left_layout.addWidget(folder_header)
        left_layout.addWidget(self.folder_button)
        left_layout.addWidget(QLabel("File List:"))
        left_layout.addWidget(self.file_list)
        left_layout.addLayout(nav_layout)
        left_widget.setLayout(left_layout)
        
        # Right panel with image view and info panel
        right_widget = QWidget()
        right_layout = QVBoxLayout()
        
        # Image and info panel splitter
        view_info_splitter = QSplitter(Qt.Vertical)
        
        # Image view panel
        image_widget = QWidget()
        image_layout = QVBoxLayout()
        
        # Create a scroll area for the image
        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setMinimumSize(600, 500)
        
        # Create the image label inside the scroll area
        self.image_label = QLabel("Select an image to display")
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.mousePressEvent = self.get_point
        self.scroll_area.setWidget(self.image_label)
        
        # Zoom controls
        zoom_layout = QHBoxLayout()
        
        self.zoom_out_button = QPushButton("-")
        self.zoom_out_button.clicked.connect(self.zoom_out)
        self.zoom_out_button.setFixedWidth(40)
        
        self.zoom_slider = QSlider(Qt.Horizontal)
        self.zoom_slider.setMinimum(int(self.parent.min_zoom * 100))
        self.zoom_slider.setMaximum(int(self.parent.max_zoom * 100))
        self.zoom_slider.setValue(int(self.parent.zoom_factor * 100))
        self.zoom_slider.valueChanged.connect(self.set_zoom)
        
        self.zoom_in_button = QPushButton("+")
        self.zoom_in_button.clicked.connect(self.zoom_in)
        self.zoom_in_button.setFixedWidth(40)
        
        self.zoom_reset_button = QPushButton("Reset Zoom")
        self.zoom_reset_button.clicked.connect(self.reset_zoom)
        
        self.zoom_label = QLabel("100%")
        self.zoom_label.setFixedWidth(60)
        
        zoom_layout.addWidget(self.zoom_out_button)
        zoom_layout.addWidget(self.zoom_slider)
        zoom_layout.addWidget(self.zoom_in_button)
        zoom_layout.addWidget(self.zoom_label)
        zoom_layout.addWidget(self.zoom_reset_button)
        
        # Action buttons
        button_layout = QHBoxLayout()
        
        self.clear_button = QPushButton("Clear Points")
        self.clear_button.clicked.connect(self.clear_points)
        
        self.prev_point_button = QPushButton("Undo Point")
        self.prev_point_button.clicked.connect(self.prev_point)
        self.prev_point_button.setToolTip("Remove the last point (Undo)")
        
        self.next_point_button = QPushButton("Redo Point")
        self.next_point_button.clicked.connect(self.next_point)
        self.next_point_button.setToolTip("Restore the last removed point (Redo)")
        
        self.fit_button = QPushButton("Fit/Unfit Ellipse")
        self.fit_button.clicked.connect(self.fit_ellipse)
        
        self.preview_button = QPushButton("Preview")
        self.preview_button.clicked.connect(self.preview_image)
        
        self.save_yolo_button = QPushButton("Save YOLO11 Labels")
        self.save_yolo_button.clicked.connect(self.save_yolo_labels)
        
        
        self.use_model_checkbox = QCheckBox("Use model")
        self.use_model_checkbox.setChecked(False)
        self.use_model_checkbox.stateChanged.connect(self.use_model_option_changed)
        
        # Class selection
        self.class_label = QLabel("Current Class:")
        self.class_combo = QComboBox()
        self.class_combo.addItem("iris (Class 0)")
        self.class_combo.addItem("pupil (Class 1)")
        self.class_combo.setCurrentIndex(0)
        self.class_combo.currentIndexChanged.connect(self.class_changed)
        
        # Object status display
        self.object_status_label = QLabel("Objects: iris(0/0) pupil(0/0)")
        
        
        self.clean_method_label = QLabel("Clean Method:")
        self.clean_method_combo = QComboBox()
        self.clean_method_combo.addItem("RANSAC")
        self.clean_method_combo.addItem("Distance Filter")
        self.clean_method_combo.addItem("Statistical")
        self.clean_method_combo.addItem("Convex Hull")
        self.clean_method_combo.addItem("None")
        self.clean_method_combo.setCurrentIndex(0)  # Default to RANSAC
        self.clean_method_combo.currentIndexChanged.connect(self.clean_method_changed)
        
        combo_layout = QHBoxLayout()
        # combo_layout.addWidget(self.method_label)
        # combo_layout.addWidget(self.method_combo)
        combo_layout.addWidget(self.clean_method_label)
        combo_layout.addWidget(self.clean_method_combo)
        
        class_layout = QHBoxLayout()
        class_layout.addWidget(self.class_label)
        class_layout.addWidget(self.class_combo)
        class_layout.addWidget(self.object_status_label)
        
        image_layout.addLayout(combo_layout)
        image_layout.addLayout(class_layout)
    
        button_layout.addWidget(self.clear_button)
        button_layout.addWidget(self.prev_point_button)
        button_layout.addWidget(self.next_point_button)
        button_layout.addWidget(self.fit_button)
        button_layout.addWidget(self.preview_button)
        button_layout.addWidget(self.save_yolo_button)
        button_layout.addWidget(self.use_model_checkbox)
        
        
        
        # Process button
        process_layout = QHBoxLayout()
        self.process_button = QPushButton("Process Labeled Images")
        self.process_button.clicked.connect(self.process_labeled_images)
        process_layout.addWidget(self.process_button)
        
        image_layout.addWidget(self.scroll_area)
        image_layout.addLayout(zoom_layout)
        image_layout.addLayout(button_layout)
        image_layout.addLayout(process_layout)
        image_widget.setLayout(image_layout)
        
        # Info panel
        info_widget = QWidget()
        info_layout = QVBoxLayout()
        
        info_header = QLabel("Information")
        
        self.info_text = QTextEdit()
        self.info_text.setReadOnly(True)
        self.info_text.setMaximumHeight(150)
        
        info_layout.addWidget(info_header)
        info_layout.addWidget(self.info_text)
        info_widget.setLayout(info_layout)
        
        # Add widgets to splitter
        view_info_splitter.addWidget(image_widget)
        view_info_splitter.addWidget(info_widget)
        view_info_splitter.setSizes([700, 100])
        
        right_layout.addWidget(view_info_splitter)
        right_widget.setLayout(right_layout)
        
        # Add widgets to content splitter
        content_splitter.addWidget(left_widget)
        content_splitter.addWidget(right_widget)
        content_splitter.setSizes([300, 1000])
        
        annotation_layout.addWidget(content_splitter)
        self.setLayout(annotation_layout)
        
        # Enable mouse wheel for zooming
        self.scroll_area.wheelEvent = self.wheel_event
        
    def clean_method_changed(self, index):
        self.current_clean_method = index
        method_names = ["RANSAC", "Distance Filter", "Statistical", "Convex Hull", "None"]
        self.update_info_text(f"Changed clean method to: {method_names[index]}")
        
        if self.use_model and self.current_image is not None:
            current_item = self.file_list.currentItem()
            if current_item:
                self.load_image(current_item)
                
    def class_changed(self, index):
        self.current_class = index
        class_name = self.class_names[index]
        self.update_info_text(f"Switched to class: {class_name} (Class {index})")
        self.update_object_status()
        self.display_image()
        
    def update_object_status(self):
        iris_points = len(self.objects[0]['points'])
        iris_ellipse = 1 if self.objects[0]['ellipse'] else 0
        pupil_points = len(self.objects[1]['points'])
        pupil_ellipse = 1 if self.objects[1]['ellipse'] else 0
        
        status_text = f"Objects: iris({iris_points}pts/{iris_ellipse}ellipse) pupil({pupil_points}pts/{pupil_ellipse}ellipse)"
        self.object_status_label.setText(status_text)
                
    def apply_styles(self):
        """Apply custom styles to the UI components"""
        # Folder and file list styles
        folder_header = self.findChild(QLabel, "Image Folder")
        if folder_header:
            folder_header.setStyleSheet(AnnotationStyles.HEADING_LABEL_STYLE)
        
        file_list_header = self.findChild(QLabel, "File List:")
        if file_list_header:
            file_list_header.setStyleSheet(AnnotationStyles.HEADING_LABEL_STYLE)
        
        self.folder_button.setStyleSheet(AnnotationStyles.BUTTON_STYLE)
        self.file_list.setStyleSheet(AnnotationStyles.LIST_WIDGET_STYLE)
        
        # Navigation buttons
        self.prev_file_button.setStyleSheet(AnnotationStyles.NAV_BUTTON_STYLE)
        self.next_file_button.setStyleSheet(AnnotationStyles.NAV_BUTTON_STYLE)
        
        # Zoom controls
        self.zoom_out_button.setStyleSheet(AnnotationStyles.ZOOM_BUTTON_STYLE)
        self.zoom_in_button.setStyleSheet(AnnotationStyles.ZOOM_BUTTON_STYLE)
        self.zoom_reset_button.setStyleSheet(AnnotationStyles.TOOL_BUTTON_STYLE)
        self.zoom_slider.setStyleSheet(AnnotationStyles.SLIDER_STYLE)
        
        # Action buttons
        self.clear_button.setStyleSheet(AnnotationStyles.DANGER_BUTTON_STYLE)
        self.prev_point_button.setStyleSheet(AnnotationStyles.LIGHT_SECONDARY_BUTTON_STYLE)
        self.next_point_button.setStyleSheet(AnnotationStyles.LIGHT_SECONDARY_BUTTON_STYLE)
        self.fit_button.setStyleSheet(AnnotationStyles.ACCENT_OUTLINE_BUTTON_STYLE)
        self.preview_button.setStyleSheet(AnnotationStyles.GRADIENT_BUTTON_STYLE)
        self.save_yolo_button.setStyleSheet(AnnotationStyles.SUCCESS_BUTTON_STYLE)
        
        # Process button
        self.process_button.setStyleSheet(AnnotationStyles.OUTLINE_BUTTON_STYLE)
        
        # Info panel
        info_header = self.findChild(QLabel, "Information")
        if info_header:
            info_header.setStyleSheet(AnnotationStyles.HEADING_LABEL_STYLE)
        
        self.info_text.setStyleSheet(AnnotationStyles.TEXT_EDIT_STYLE)
        
        # Image label
        self.image_label.setStyleSheet("QLabel { background-color: #F0F0F0; border: 1px solid #CCCCCC; }")
        
        # Scroll area
        self.scroll_area.setStyleSheet(AnnotationStyles.SCROLL_AREA_STYLE + AnnotationStyles.SCROLLBAR_STYLE)
        
        # Splitters
        for splitter in self.findChildren(QSplitter):
            splitter.setStyleSheet(AnnotationStyles.SPLITTER_STYLE)
            
        
        # self.method_label.setStyleSheet(AnnotationStyles.HEADING_LABEL_STYLE)
        self.clean_method_label.setStyleSheet(AnnotationStyles.HEADING_LABEL_STYLE)
        
        combobox_style = """
            QComboBox {
                background-color: #F4F4F4;
                border: 1px solid #CCCCCC;
                border-radius: 4px;
                padding: 3px 18px 3px 8px;
                min-width: 150px;
                color: #333333;
            }
            QComboBox:hover {
                border: 1px solid #5A8DEE;
            }
            QComboBox::drop-down {
                subcontrol-origin: padding;
                subcontrol-position: top right;
                width: 20px;
                border-left: 1px solid #CCCCCC;
            }
        """
        
        # self.method_combo.setStyleSheet(combobox_style)
        self.clean_method_combo.setStyleSheet(combobox_style)
        
        # Class selection styles
        self.class_label.setStyleSheet(AnnotationStyles.HEADING_LABEL_STYLE)
        self.class_combo.setStyleSheet(combobox_style)
        self.object_status_label.setStyleSheet("color: #666666; font-weight: bold;")
         
    def use_model_option_changed(self, state):
        if state == Qt.Checked:
            self.use_model = True
        else:
            self.use_model = False
            
    def wheel_event(self, event):
        delta = event.angleDelta().y()
        if delta > 0:
            self.zoom_in()
        else:
            self.zoom_out()
        event.accept()
    
    def zoom_in(self):
        if self.parent.zoom_factor < self.parent.max_zoom:
            self.parent.zoom_factor += 0.1
            self.update_zoom()
    
    def zoom_out(self):
        if self.parent.zoom_factor > self.parent.min_zoom:
            self.parent.zoom_factor -= 0.1
            self.update_zoom()
    
    def reset_zoom(self):
        self.parent.zoom_factor = 1.0
        self.update_zoom()
    
    def set_zoom(self, value):
        self.parent.zoom_factor = value / 100
        self.update_zoom(from_slider=True)
    
    def update_zoom(self, from_slider=False):
        if self.current_image is not None:
            self.parent.zoom_factor = max(self.parent.min_zoom, min(self.parent.max_zoom, self.parent.zoom_factor))
            
            if not from_slider:
                self.zoom_slider.setValue(int(self.parent.zoom_factor * 100))
            
            self.zoom_label.setText(f"{int(self.parent.zoom_factor * 100)}%")
            self.display_image()
    
    def select_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "Select Image Folder")
        if folder:
            self.parent.current_folder = folder
            
            self.yolo_segment_dir = os.path.join(self.parent.current_folder, "yolo_segment_labels")
            self.ellipse_label_dir = os.path.join(self.parent.current_folder, "ellipse_labels")
            
            os.makedirs(self.yolo_segment_dir, exist_ok=True)
            os.makedirs(self.ellipse_label_dir, exist_ok=True)
            
            self.load_file_list()
            self.update_info_text(f"Loaded folder: {folder}")
    
    def load_file_list(self):
        self.file_list.clear()
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif']
        
        labeled_files = set()
        for label_file in os.listdir(self.yolo_segment_dir):
            if label_file.endswith('.txt'):
                base_name = os.path.splitext(label_file)[0]
                labeled_files.add(base_name)
        
        for file in os.listdir(self.parent.current_folder):
            file_path = os.path.join(self.parent.current_folder, file)
            if os.path.isfile(file_path):
                ext = os.path.splitext(file)[1].lower()
                if ext in image_extensions:
                    item = QListWidgetItem(file)
                    
                    base_name = os.path.splitext(file)[0]
                    if base_name in labeled_files:
                        item.setIcon(create_checkmark_icon())
                    
                    self.file_list.addItem(item)
    
    def check_for_existing_labels(self, file_name):
        base_name = os.path.splitext(file_name)[0]
        segment_label_path = os.path.join(self.yolo_segment_dir, f"{base_name}.txt")
        ellipse_label_path = os.path.join(self.ellipse_label_dir, f"{base_name}.txt")
        
        # Reset objects data
        self.objects = {0: {'points': [], 'ellipse': None}, 1: {'points': [], 'ellipse': None}}
        self.points = []  # Legacy support
        self.ellipse_params = None  # Legacy support
        
        # Load segmentation labels
        if os.path.exists(segment_label_path):
            with open(segment_label_path, 'r') as f:
                lines = f.read().strip().split('\n')
            
            img_height, img_width = self.current_image.shape[:2]
            
            for line in lines:
                if not line.strip():
                    continue
                    
                parts = line.split()
                if len(parts) > 1:
                    class_id = int(parts[0])
                    points_data = parts[1:]
                    
                    if class_id in [0, 1]:  # Valid class IDs
                        points = []
                        for i in range(0, len(points_data), 2):
                            if i + 1 < len(points_data):
                                x = float(points_data[i]) * img_width
                                y = float(points_data[i + 1]) * img_height
                                points.append((int(x), int(y)))
                        
                        self.objects[class_id]['points'] = points
                        
                        # Legacy support - use first found object
                        if not self.points:
                            self.points = points.copy()
        
        # Load ellipse labels
        if os.path.exists(ellipse_label_path):
            with open(ellipse_label_path, 'r') as f:
                lines = f.read().strip().split('\n')
            
            img_height, img_width = self.current_image.shape[:2]
            
            for line in lines:
                if not line.strip():
                    continue
                    
                parts = line.split()
                if len(parts) >= 6:
                    class_id = int(parts[0])
                    
                    if class_id in [0, 1]:  # Valid class IDs
                        center_x = float(parts[1]) * img_width
                        center_y = float(parts[2]) * img_height
                        axes_x = float(parts[3]) * img_width
                        axes_y = float(parts[4]) * img_height
                        angle = float(parts[5])
                        
                        ellipse = ((center_x, center_y), (axes_x, axes_y), angle)
                        self.objects[class_id]['ellipse'] = ellipse
                        
                        # Legacy support - use first found ellipse
                        if not self.ellipse_params:
                            self.ellipse_params = ellipse
        
        # Check if any data was loaded
        has_data = False
        for class_id in [0, 1]:
            if len(self.objects[class_id]['points']) > 0 or self.objects[class_id]['ellipse'] is not None:
                has_data = True
                break
                
        return has_data
    
    def load_image(self, item):
        if not item:
            return
            
        # Reset multi-object data
        self.objects = {0: {'points': [], 'ellipse': None}, 1: {'points': [], 'ellipse': None}}
        self.deleted_points = {0: [], 1: []}
        
        # Legacy support
        self.points = []
        self.ellipse_params = None
        
        file_name = item.text()
        self.current_image_path = os.path.join(self.parent.current_folder, file_name)
        self.current_image = cv2.imread(self.current_image_path)
        
        has_labels = self.check_for_existing_labels(file_name)
        
        if self.use_model:
            x, y, w, h = self.crop_rect
            current_image_copy = self.current_image.copy()
            cropped_img = current_image_copy[y:y+h, x:x+w]
            
            if self.segment_type == "instance":
                cropped_img = cv2.cvtColor(cropped_img, cv2.COLOR_BGR2RGB)
                cv2.imwrite("test.png", cropped_img)
                predicts = self.model.predict(cropped_img)
                if len(predicts[0].masks) > 0:
                    for i, mask in enumerate(predicts[0].masks):
                        # Get contour points from the crop
                        contour_points = mask.xy[0]
                        contour_points_cv = contour_points.astype(np.int32)
                        
                        clean_methods = ['ransac', 'distance_filter', 'statistical', 'convex_hull', None]
                        selected_method = clean_methods[self.current_clean_method]
                        
                        if selected_method:
                            cleaned_contour_points = clean_contour_points(contour_points_cv, method=selected_method)
                        else:
                            cleaned_contour_points = contour_points_cv
                            
                        # Convert contour points to original image coordinates
                        original_contour_points = cleaned_contour_points.copy()
                        original_contour_points[:, 0] += x  # Add x offset to all x-coordinates
                        original_contour_points[:, 1] += y  # Add y offset to all y-coordinates
                        
                        print("Original contour points:", original_contour_points)
                        
                        if len(cleaned_contour_points) >= 5:
                            if self.current_method == 0:  # Standard
                                ellipse = fit_ellipse_standard(cleaned_contour_points)
                            elif self.current_method == 1:  # Direct Least Squares
                                ellipse = fit_ellipse_direct(cleaned_contour_points)
                            elif self.current_method == 2:  # EllipseModel from scikit-image
                                ellipse = fit_ellipse_skimage(cleaned_contour_points)

                            # Convert ellipse center to original image coordinates
                            original_center = (ellipse['center'][0] + x, ellipse['center'][1] + y)
                            
                            # Save the ellipse parameters with original coordinates
                            self.ellipse_params = (original_center, ellipse['axes'], ellipse['angle'])
                            # Store in current class object
                self.objects[self.current_class]['points'] = original_contour_points.tolist()
                self.objects[self.current_class]['ellipse'] = (original_center, ellipse['axes'], ellipse['angle'])
                
                # Legacy support
                self.points = original_contour_points.tolist()
                self.ellipse_params = (original_center, ellipse['axes'], ellipse['angle'])
                            
                print("Original ellipse:", {
                    'center': original_center,
                    'axes': ellipse['axes'],
                    'angle': ellipse['angle']
                })
            else: 
                pass
        
        else:
            if has_labels:
                # Show loaded objects info
                iris_points = len(self.objects[0]['points'])
                pupil_points = len(self.objects[1]['points'])
                iris_ellipse = self.objects[0]['ellipse'] is not None
                pupil_ellipse = self.objects[1]['ellipse'] is not None
                
                info_text = f"Loaded image: {file_name} with existing labels\n"
                info_text += f"Iris: {iris_points} points, {'ellipse' if iris_ellipse else 'no ellipse'}\n"
                info_text += f"Pupil: {pupil_points} points, {'ellipse' if pupil_ellipse else 'no ellipse'}"
                
                self.update_info_text(info_text)
                self.update_object_status()
            else:
                self.update_info_text(f"Loaded image: {file_name}\nNo existing labels found")
                self.update_object_status()
            
        self.display_image()
    
    def display_image(self, flag_show=True, flag_ellipse=True):
        if self.current_image is None:
            return
        
        display_image = self.current_image.copy()
        
        if flag_show:
            # Draw objects for all classes
            class_colors = [(0, 0, 255), (255, 0, 0)]  # Red for iris, Blue for pupil
            class_labels = ['I', 'P']  # Short labels for iris and pupil
            
            for class_id in [0, 1]:
                points = self.objects[class_id]['points']
                ellipse = self.objects[class_id]['ellipse']
                color = class_colors[class_id]
                
                # Draw points
                for i, point in enumerate(points):
                    point_tuple = (int(point[0]), int(point[1])) if isinstance(point, (list, tuple, np.ndarray)) else None
                    
                    if point_tuple is None:
                        continue
                    
                    # Highlight current class with bigger size
                    size = 4 if class_id == self.current_class else 2
                    cv2.circle(display_image, point_tuple, size, color, -1)
                    
                    # Label with class and point number
                    label = f"{class_labels[class_id]}{i+1}"
                    cv2.putText(display_image, label, 
                                (point_tuple[0]+5, point_tuple[1]-5), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
                
                # Draw ellipse if available
                if ellipse is not None and flag_ellipse:
                    center, axes, _ = ellipse
                    center = tuple(map(int, center))
                    axes = tuple(map(int, axes))
                    
                    # Draw the ellipse with thicker line for current class
                    thickness = 3 if class_id == self.current_class else 2
                    cv2.ellipse(display_image, ellipse, color, thickness)
                    
                    # Draw center point
                    cv2.circle(display_image, center, 5, color, -1)
            
            # Legacy support - also draw old points/ellipse if they exist
            for i, point in enumerate(self.points):
                point_tuple = (int(point[0]), int(point[1])) if isinstance(point, (list, tuple, np.ndarray)) else None
                
                if point_tuple is None:
                    continue
                
                # Draw legacy points in yellow
                cv2.circle(display_image, point_tuple, 3, (0, 255, 255), -1)
                cv2.putText(display_image, str(i+1), 
                            (point_tuple[0]+5, point_tuple[1]-5), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
            
            # Draw legacy ellipse
            if self.ellipse_params is not None and flag_ellipse:
                center, axes, _ = self.ellipse_params
                center = tuple(map(int, center))
                axes = tuple(map(int, axes))
                
                cv2.ellipse(display_image, self.ellipse_params, (0, 255, 255), 2)
                cv2.circle(display_image, center, 5, (0, 255, 255), -1)
                
                # angle_rad = np.radians(angle)
                
                # # Calculate line length based on the ellipse major axis
                # line_length = max(axes[0], axes[1]) * 1.5
                

                # direction_angle_rad = angle_rad + np.radians(90)
                
                # end_x = int(center[0] + line_length * np.cos(direction_angle_rad))
                # end_y = int(center[1] + line_length * np.sin(direction_angle_rad))
                
                # Draw the direction line (yellow, thicker)
                # cv2.line(display_image, center, (end_x, end_y), (0, 255, 255), 1)
                
                # Add an arrowhead to make direction clearer
                # arrowhead_length = 15
                # arrowhead_angle1 = direction_angle_rad + np.radians(150)
                # arrowhead_angle2 = direction_angle_rad - np.radians(150)
                
                # arrow_x1 = int(end_x - arrowhead_length * np.cos(arrowhead_angle1))
                # arrow_y1 = int(end_y - arrowhead_length * np.sin(arrowhead_angle1))
                # arrow_x2 = int(end_x - arrowhead_length * np.cos(arrowhead_angle2))
                # arrow_y2 = int(end_y - arrowhead_length * np.sin(arrowhead_angle2))
                
                # cv2.line(display_image, (end_x, end_y), (arrow_x1, arrow_y1), (0, 255, 255), 2)
                # cv2.line(display_image, (end_x, end_y), (arrow_x2, arrow_y2), (0, 255, 255), 2)
        
        height, width, _ = display_image.shape
        bytes_per_line = 3 * width
        
        
        display_image = cv2.cvtColor(display_image, cv2.COLOR_BGR2RGB)
        q_image = QImage(display_image.data, width, height, bytes_per_line, QImage.Format_RGB888)
        
        pixmap = QPixmap.fromImage(q_image)
        
        if self.crop_rect:
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
        
        self.image_label.setPixmap(pixmap)
        self.image_label.setFixedSize(pixmap.size())

    def update_info_text(self, text, append=False):
        if append:
            current_text = self.info_text.toPlainText()
            self.info_text.setPlainText(f"{current_text}\n{text}")
        else:
            self.info_text.setPlainText(text)
        self.parent.status_bar.showMessage(text.split('\n')[0])
    
    def get_point(self, event):
        if self.current_image is None:
            return
        
        label_pos = event.pos()
        
        pixmap = self.image_label.pixmap()
        if pixmap:
            img_height, img_width = self.current_image.shape[:2]
            
            # Account for zoom factor
            x = int(label_pos.x() / self.parent.zoom_factor)
            y = int(label_pos.y() / self.parent.zoom_factor)
            
            if 0 <= x < img_width and 0 <= y < img_height:
                # Add point to current class
                current_points = self.objects[self.current_class]['points']
                current_points.append((x, y))
                
                # Clear redo history for current class when adding new points
                self.deleted_points[self.current_class] = []
                
                # Update legacy points (for backward compatibility)
                self.points.append((x, y))
                
                self.display_image()
                class_name = self.class_names[self.current_class]
                self.update_info_text(f"Added {class_name} point {len(current_points)} at ({x}, {y})")
                self.update_object_status()
                
                # Auto fit ellipse when we have 5 or more points for current class
                if len(current_points) >= 5:
                    self.fit_ellipse()
    
    def clear_points(self):
        current_points = self.objects[self.current_class]['points']
        current_ellipse = self.objects[self.current_class]['ellipse']
        
        if not current_points and not current_ellipse:
            return
        
        # Save current points to deleted_points for possible redo
        if current_points:
            self.deleted_points[self.current_class] = current_points.copy()
        else:
            self.deleted_points[self.current_class] = []
            
        # Clear current class data
        self.objects[self.current_class]['points'] = []
        self.objects[self.current_class]['ellipse'] = None
        
        # Legacy support
        self.points = []
        self.ellipse_params = None
        
        self.display_image()
        class_name = self.class_names[self.current_class]
        self.update_info_text(f"Cleared {class_name} points and ellipse")
        self.update_object_status()
    
    def prev_point(self):
        """Remove the last added point (undo)"""
        current_points = self.objects[self.current_class]['points']
        
        if not current_points:
            class_name = self.class_names[self.current_class]
            self.update_info_text(f"No {class_name} points to undo")
            return
        
        removed_point = current_points.pop()
        self.deleted_points[self.current_class].append(removed_point)
        
        # Update legacy points
        if self.points:
            self.points.pop()
        
        if len(current_points) >= 5:
            self.fit_ellipse()
        else:
            # Clear ellipse if not enough points
            self.objects[self.current_class]['ellipse'] = None
        
        self.display_image()
        class_name = self.class_names[self.current_class]
        self.update_info_text(f"Removed last {class_name} point. {len(current_points)} points remaining")
        self.update_object_status()
    
    def next_point(self):
        """Restore the last deleted point (redo)"""
        if not self.deleted_points[self.current_class]:
            class_name = self.class_names[self.current_class]
            self.update_info_text(f"No {class_name} points to redo")
            return
        
        restored_point = self.deleted_points[self.current_class].pop()
        self.objects[self.current_class]['points'].append(restored_point)
        
        # Update legacy points
        self.points.append(restored_point)
        
        current_points = self.objects[self.current_class]['points']
        if len(current_points) >= 5:
            self.fit_ellipse()

        self.display_image()
        class_name = self.class_names[self.current_class]
        self.update_info_text(f"Restored {class_name} point. Now {len(current_points)} points")
        self.update_object_status()
    
    def prev_file(self):
        current_row = self.file_list.currentRow()
        if current_row > 0:
            self.file_list.setCurrentRow(current_row - 1)
            self.load_image(self.file_list.currentItem())
    
    def next_file(self):
        current_row = self.file_list.currentRow()
        if current_row < self.file_list.count() - 1:
            self.file_list.setCurrentRow(current_row + 1)
            self.load_image(self.file_list.currentItem())
    
    def fit_ellipse(self):
        current_points = self.objects[self.current_class]['points']
        
        if len(current_points) < 5:
            class_name = self.class_names[self.current_class]
            QMessageBox.warning(self, "Warning", f"At least 5 points are needed to create an ellipse for {class_name}!")
            return
        
        try:
            self.flag_ellipse = not self.flag_ellipse
            points_array = np.array(current_points, dtype=np.int32)
            
            ellipse_params = cv2.fitEllipse(points_array)
            
            # Store ellipse for current class
            self.objects[self.current_class]['ellipse'] = ellipse_params
            
            # Legacy support
            self.ellipse_params = ellipse_params
            
            center, axes, angle = ellipse_params
            
            class_name = self.class_names[self.current_class]
            info_text = f"{class_name} ellipse fitted successfully\n" \
                    f"Center: ({center[0]:.1f}, {center[1]:.1f})\n" \
                    f"Axes: ({axes[0]:.1f}, {axes[1]:.1f})\n" \
                    f"Angle: {angle:.1f} degrees\n" \
                    f"(Angle represents rotation of the ellipse from horizontal axis)"
                    
            self.update_info_text(info_text)
            self.update_object_status()
            self.display_image(flag_ellipse=self.flag_ellipse)
            
        except Exception as e:
            class_name = self.class_names[self.current_class]
            error_msg = f"Cannot create {class_name} ellipse: {str(e)}"
            QMessageBox.critical(self, "Error", error_msg)
            self.update_info_text(error_msg)
    
    def preview_image(self):
        self.flag_preview = not self.flag_preview
        self.display_image(self.flag_preview)
    
    def save_yolo_labels(self):
        if self.current_image is None:
            QMessageBox.warning(self, "Warning", "No image loaded!")
            return
            
        # Check if any objects have points
        has_data = False
        for class_id in [0, 1]:
            if len(self.objects[class_id]['points']) >= 3:
                has_data = True
                break
                
        if not has_data:
            QMessageBox.warning(self, "Warning", "Not enough points for any object!")
            return
            
        try:
            # Get image dimensions
            img_height, img_width = self.current_image.shape[:2]
            
            filename = os.path.splitext(os.path.basename(self.current_image_path))[0]
            yolo_label_path = os.path.join(self.yolo_segment_dir, f"{filename}.txt")
            ellipse_label_path = os.path.join(self.ellipse_label_dir, f"{filename}.txt")
            
            # Prepare YOLO segmentation lines for multiple objects
            yolo_lines = []
            ellipse_lines = []
            
            saved_objects = []
            
            for class_id in [0, 1]:  # iris=0, pupil=1
                points = self.objects[class_id]['points']
                ellipse = self.objects[class_id]['ellipse']
                
                if len(points) >= 3:
                    # Normalize points for YOLO format
                    normalized_points = []
                    for point in points:
                        x_norm = point[0] / img_width
                        y_norm = point[1] / img_height
                        normalized_points.append(x_norm)
                        normalized_points.append(y_norm)
                    
                    # Create YOLO segmentation format
                    yolo_line = f"{class_id} " + " ".join([f"{p:.15f}" for p in normalized_points])
                    yolo_lines.append(yolo_line)
                    
                    saved_objects.append(f"{self.class_names[class_id]}({len(points)}pts)")
                    
                # Save ellipse parameters if available
                if ellipse is not None:
                    center, axes, angle = ellipse
                    
                    # Normalize ellipse parameters
                    center_x_norm = center[0] / img_width
                    center_y_norm = center[1] / img_height
                    axes_x_norm = axes[0] / img_width
                    axes_y_norm = axes[1] / img_height
                    
                    # Format: class_id center_x center_y axes_x axes_y angle
                    ellipse_line = f"{class_id} {center_x_norm:.15f} {center_y_norm:.15f} {axes_x_norm:.15f} {axes_y_norm:.15f} {angle:.15f}"
                    ellipse_lines.append(ellipse_line)
            
            # Save segmentation labels
            if yolo_lines:
                with open(yolo_label_path, 'w') as f:
                    f.write('\n'.join(yolo_lines))
            
            # Save ellipse parameters
            if ellipse_lines:
                with open(ellipse_label_path, 'w') as f:
                    f.write('\n'.join(ellipse_lines))
            
            # Update info
            objects_str = ", ".join(saved_objects)
            ellipse_count = len(ellipse_lines)
            
            info_message = f"Labels saved for {filename}\n" + \
                          f"Objects: {objects_str}\n" + \
                          f"Ellipses: {ellipse_count}"
            
            self.update_info_text(info_message)
            
            # Update icon in file list
            current_row = self.file_list.currentRow()
            if current_row >= 0:
                item = self.file_list.item(current_row)
                if item.icon().isNull():
                    item.setIcon(create_checkmark_icon())
                
        except Exception as e:
            error_msg = f"Failed to save labels: {str(e)}"
            QMessageBox.critical(self, "Error", error_msg)
            self.update_info_text(error_msg)
            
    def process_labeled_images(self):
        if not self.parent.current_folder:
            QMessageBox.warning(self, "Warning", "No folder selected!")
            return
            
        try:
            # Create data directory structure
            data_dir = os.path.join(self.parent.current_folder, "data")
            images_dir = os.path.join(data_dir, "images")
            labels_dir = os.path.join(data_dir, "labels")
            labels_ellipse_dir = os.path.join(data_dir, "labels_ellipse")
            
            os.makedirs(images_dir, exist_ok=True)
            os.makedirs(labels_dir, exist_ok=True)
            os.makedirs(labels_ellipse_dir, exist_ok=True)
            
            # Get the folder name (last part of the path)
            folder_name = os.path.basename(os.path.normpath(self.parent.current_folder))
            
            # Get list of labeled files
            labeled_files = set()
            for label_file in os.listdir(self.yolo_segment_dir):
                if label_file.endswith('.txt'):
                    base_name = os.path.splitext(label_file)[0]
                    labeled_files.add(base_name)
            
            if not labeled_files:
                QMessageBox.warning(self, "Warning", "No labeled images found!")
                return
            
            # Process each labeled file
            processed_count = 0
            for base_name in labeled_files:
                # Check original file pattern (timestamp_x_y)
                parts = base_name.split('_')
                if len(parts) >= 3:
                    # Original filename parts
                    timestamp = parts[0]
                    x_coord = parts[1]
                    y_coord = parts[2]
                else:
                    timestamp = parts[1]
                    x_coord = 0
                    y_coord = 0
                    
                # New filename format: foldername_timestamp_x_y
                new_base_name = f"{folder_name}_{timestamp}_{x_coord}_{y_coord}"
                
                # Source files
                image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif']
                image_file = None
                
                for ext in image_extensions:
                    potential_file = os.path.join(self.parent.current_folder, f"{base_name}{ext}")
                    if os.path.exists(potential_file):
                        image_file = potential_file
                        break
                
                segment_file = os.path.join(self.yolo_segment_dir, f"{base_name}.txt")
                ellipse_file = os.path.join(self.ellipse_label_dir, f"{base_name}.txt")
                
                # Destination files
                if image_file:
                    img_ext = os.path.splitext(image_file)[1]
                    new_image_file = os.path.join(images_dir, f"{new_base_name}{img_ext}")
                    new_segment_file = os.path.join(labels_dir, f"{new_base_name}.txt")
                    new_ellipse_file = os.path.join(labels_ellipse_dir, f"{new_base_name}.txt")
                    
                    # Copy files with new names
                    import shutil
                    shutil.copy2(image_file, new_image_file)
                    
                    if os.path.exists(segment_file):
                        shutil.copy2(segment_file, new_segment_file)
                    
                    if os.path.exists(ellipse_file):
                        shutil.copy2(ellipse_file, new_ellipse_file)
                        
                    processed_count += 1
            
            success_message = f"Processed {processed_count} labeled images.\n" \
                             f"Files saved to {data_dir} with new naming format:\n" \
                             f"{folder_name}_timestamp_x_y"
            
            self.update_info_text(success_message)
            QMessageBox.information(self, "Success", success_message)
            
        except Exception as e:
            error_msg = f"Error processing labeled images: {str(e)}"
            QMessageBox.critical(self, "Error", error_msg)
            self.update_info_text(error_msg)
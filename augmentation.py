from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel, 
                             QPushButton, QFileDialog, QMessageBox, QListWidget,
                             QComboBox, QSpinBox, QDoubleSpinBox, QCheckBox, QGroupBox, 
                             QGridLayout, QScrollArea, QSplitter, QProgressBar, QDialog)
from PyQt5.QtGui import QPixmap, QImage, QPainter, QPen, QColor
from PyQt5.QtCore import Qt, QThread, pyqtSignal
import os
import cv2
import numpy as np
import random
import albumentations as A

from styles import AnnotationStyles

class AugmentationThread(QThread):
    """Thread for running augmentation in background"""
    progress = pyqtSignal(int)
    completed = pyqtSignal(str)
    error = pyqtSignal(str)
    
    def __init__(self, augmentor, dataset, image_files, output_dir, num_augmentations):
        super().__init__()
        self.augmentor = augmentor
        self.dataset = dataset
        self.image_files = image_files
        self.output_dir = output_dir
        self.num_augmentations = num_augmentations
        
    def run(self):
        try:
            total_files = len(self.image_files)
            processed = 0
            
            for image_file in self.image_files:
                self.augmentor.augment_single_image(
                    image_file, 
                    self.dataset, 
                    self.output_dir, 
                    self.num_augmentations
                )
                processed += 1
                progress_percent = int((processed / total_files) * 100)
                self.progress.emit(progress_percent)
            
            summary = f"Generated {total_files * self.num_augmentations} augmented images from {total_files} original images"
            self.completed.emit(summary)
            
        except Exception as e:
            self.error.emit(f"Error during augmentation: {str(e)}")

class AugmentationTab(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.parent = parent
        
        # Initialize properties
        self.augmentation_folder = None
        self.current_image_path = None
        self.current_image = None
        self.current_segment = None
        self.augmented_image = None
        self.current_image_index = 0
        self.bboxes = []
        self.keypoints = []
        self.image_files = []
        
        # Setup the UI
        self.setup_ui()
        
        self.apply_styles()
    
    def setup_ui(self):
        layout = QVBoxLayout()
        
        # Top section - folder selection
        top_layout = QHBoxLayout()
        self.augmentation_folder_button = QPushButton("Select Data Folder")
        self.augmentation_folder_button.clicked.connect(self.select_augmentation_folder)
        
        self.selected_folder_label = QLabel("No folder selected")
        
        self.dataset_combo = QComboBox()
        self.dataset_combo.addItems(["train", "test", "valid"])
        self.dataset_combo.currentIndexChanged.connect(self.load_dataset)
        
        top_layout.addWidget(self.augmentation_folder_button)
        top_layout.addWidget(self.selected_folder_label, 1)
        top_layout.addWidget(QLabel("Dataset:"))
        top_layout.addWidget(self.dataset_combo)
        
        # Main content with options and preview
        content_splitter = QSplitter(Qt.Horizontal)
        
        # Left panel - file list and augmentation options
        left_widget = QWidget()
        left_layout = QVBoxLayout()
        
        # File list panel
        file_widget = QWidget()
        file_layout = QVBoxLayout()
        file_header = QLabel("Image Files")
        file_layout.addWidget(file_header)
        
        self.file_list = QListWidget()
        self.file_list.itemClicked.connect(self.load_image)
        
        file_layout.addWidget(self.file_list)
        file_widget.setLayout(file_layout)
        
        # Transformation options group
        transform_group = QGroupBox("Transformation Options")
        transform_layout = QVBoxLayout()
        
        # Horizontal Flip Row
        hflip_row = QHBoxLayout()
        self.hflip_check = QCheckBox("Horizontal Flip")
        self.hflip_check.setChecked(True)
        self.hflip_prob = QDoubleSpinBox()
        self.hflip_prob.setRange(0.0, 1.0)
        self.hflip_prob.setValue(0.5)
        self.hflip_prob.setSingleStep(0.1)
        self.hflip_prob.setDecimals(1)
        
        hflip_row.addWidget(self.hflip_check)
        hflip_row.addWidget(QLabel("Probability:"))
        hflip_row.addWidget(self.hflip_prob)
        transform_layout.addLayout(hflip_row)
        
        # Rotation Row
        rotate_row = QHBoxLayout()
        self.rotate_check = QCheckBox("Rotation")
        self.rotate_check.setChecked(True)
        self.rotate_prob = QDoubleSpinBox()
        self.rotate_prob.setRange(0.0, 1.0)
        self.rotate_prob.setValue(0.5)
        self.rotate_prob.setSingleStep(0.1)
        self.rotate_prob.setDecimals(1)
        
        self.rotate_limit = QSpinBox()
        self.rotate_limit.setRange(0, 180)
        self.rotate_limit.setValue(10)
        self.rotate_limit.setSuffix("°")
        
        rotate_row.addWidget(self.rotate_check)
        rotate_row.addWidget(QLabel("Probability:"))
        rotate_row.addWidget(self.rotate_prob)
        rotate_row.addWidget(QLabel("Limit:"))
        rotate_row.addWidget(self.rotate_limit)
        transform_layout.addLayout(rotate_row)
        
        # Blur Row
        blur_row = QHBoxLayout()
        self.blur_check = QCheckBox("Blur")
        self.blur_check.setChecked(True)
        self.blur_prob = QDoubleSpinBox()
        self.blur_prob.setRange(0.0, 1.0)
        self.blur_prob.setValue(0.3)
        self.blur_prob.setSingleStep(0.1)
        self.blur_prob.setDecimals(1)
        
        self.blur_limit = QSpinBox()
        self.blur_limit.setRange(1, 15)
        self.blur_limit.setValue(3)
        
        blur_row.addWidget(self.blur_check)
        blur_row.addWidget(QLabel("Probability:"))
        blur_row.addWidget(self.blur_prob)
        blur_row.addWidget(QLabel("Kernel Size:"))
        blur_row.addWidget(self.blur_limit)
        transform_layout.addLayout(blur_row)
        
        transform_group.setLayout(transform_layout)
        
        # Crop options group
        crop_group = QGroupBox("Random Crop")
        crop_layout = QVBoxLayout()
        
        # Crop Row
        crop_row = QHBoxLayout()
        self.crop_check = QCheckBox("Random Crop")
        self.crop_check.setChecked(True)
        self.crop_prob = QDoubleSpinBox()
        self.crop_prob.setRange(0.0, 1.0)
        self.crop_prob.setValue(0.5)
        self.crop_prob.setSingleStep(0.1)
        self.crop_prob.setDecimals(1)
        
        crop_row.addWidget(self.crop_check)
        crop_row.addWidget(QLabel("Probability:"))
        crop_row.addWidget(self.crop_prob)
        crop_layout.addLayout(crop_row)
        
        # Crop Size Rows
        crop_height_row = QHBoxLayout()
        self.crop_min_height_label = QLabel("Min Height (%):")
        self.crop_min_height = QSpinBox()
        self.crop_min_height.setRange(10, 100)
        self.crop_min_height.setValue(80)
        self.crop_min_height.setSuffix("%")
        
        crop_height_row.addWidget(self.crop_min_height_label)
        crop_height_row.addWidget(self.crop_min_height)
        crop_layout.addLayout(crop_height_row)
        
        crop_width_row = QHBoxLayout()
        self.crop_min_width_label = QLabel("Min Width (%):")
        self.crop_min_width = QSpinBox()
        self.crop_min_width.setRange(10, 100)
        self.crop_min_width.setValue(80)
        self.crop_min_width.setSuffix("%")
        
        crop_width_row.addWidget(self.crop_min_width_label)
        crop_width_row.addWidget(self.crop_min_width)
        crop_layout.addLayout(crop_width_row)
        
        crop_group.setLayout(crop_layout)
        
        # Noise options group
        noise_group = QGroupBox("Noise")
        noise_layout = QVBoxLayout()
        
        # Noise Row
        noise_row = QHBoxLayout()
        self.noise_check = QCheckBox("Add Noise")
        self.noise_check.setChecked(True)
        self.noise_prob = QDoubleSpinBox()
        self.noise_prob.setRange(0.0, 1.0)
        self.noise_prob.setValue(0.5)
        self.noise_prob.setSingleStep(0.1)
        self.noise_prob.setDecimals(1)
        
        noise_row.addWidget(self.noise_check)
        noise_row.addWidget(QLabel("Probability:"))
        noise_row.addWidget(self.noise_prob)
        noise_layout.addLayout(noise_row)
        
        # Noise Type Row
        noise_type_row = QHBoxLayout()
        self.noise_type_label = QLabel("Noise Type:")
        self.noise_type_combo = QComboBox()
        self.noise_type_combo.addItems(["gaussian", "poisson", "salt", "pepper", "s&p"])
        
        noise_type_row.addWidget(self.noise_type_label)
        noise_type_row.addWidget(self.noise_type_combo)
        noise_layout.addLayout(noise_type_row)
        
        # Noise Amount Row
        noise_amount_row = QHBoxLayout()
        self.noise_amount_label = QLabel("Amount:")
        self.noise_amount = QDoubleSpinBox()
        self.noise_amount.setRange(0.01, 1.0)
        self.noise_amount.setValue(0.1)
        self.noise_amount.setSingleStep(0.05)
        self.noise_amount.setDecimals(2)
        
        noise_amount_row.addWidget(self.noise_amount_label)
        noise_amount_row.addWidget(self.noise_amount)
        noise_layout.addLayout(noise_amount_row)
        
        noise_group.setLayout(noise_layout)
        
        # Generation settings
        settings_group = QGroupBox("Generation Settings")
        settings_layout = QVBoxLayout()
        
        # Number of augmentations per image
        num_aug_row = QHBoxLayout()
        self.num_augmentations_label = QLabel("Augmentations per image:")
        self.num_augmentations = QSpinBox()
        self.num_augmentations.setRange(1, 100)
        self.num_augmentations.setValue(3)
        num_aug_row.addWidget(self.num_augmentations_label)
        num_aug_row.addWidget(self.num_augmentations)
        settings_layout.addLayout(num_aug_row)
        
        # Random seed
        seed_row = QHBoxLayout()
        self.random_seed_check = QCheckBox("Fixed random seed")
        self.random_seed_check.setChecked(False)
        self.random_seed = QSpinBox()
        self.random_seed.setRange(0, 9999)
        self.random_seed.setValue(42)
        self.random_seed.setEnabled(False)
        self.random_seed_check.stateChanged.connect(
            lambda state: self.random_seed.setEnabled(state == Qt.Checked)
        )
        seed_row.addWidget(self.random_seed_check)
        seed_row.addWidget(self.random_seed)
        settings_layout.addLayout(seed_row)
        
        settings_group.setLayout(settings_layout)
        
        # Generate button and progress bar
        gen_layout = QVBoxLayout()
        
        self.generate_button = QPushButton("Generate Augmented Data")
        self.generate_button.setEnabled(False)
        self.generate_button.clicked.connect(self.generate_augmented_data)
        
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        
        gen_layout.addWidget(self.generate_button)
        gen_layout.addWidget(self.progress_bar)
        
        # Add components to left layout
        left_layout.addWidget(file_widget)
        left_layout.addWidget(transform_group)
        left_layout.addWidget(crop_group)
        left_layout.addWidget(noise_group)
        left_layout.addWidget(settings_group)
        left_layout.addLayout(gen_layout)
        left_layout.addStretch()
        left_widget.setLayout(left_layout)
        
        # Right panel - preview and info
        right_widget = QWidget()
        right_layout = QVBoxLayout()
        
        # Preview panel
        preview_widget = QWidget()
        preview_layout = QVBoxLayout()
        preview_header = QLabel("Preview Augmentation")
        preview_layout.addWidget(preview_header)
        
        self.preview_scroll_area = QScrollArea()
        self.preview_scroll_area.setWidgetResizable(True)
        self.preview_scroll_area.setMinimumHeight(300)
        
        self.preview_label = QLabel("Select an image and click 'Generate Preview'")
        self.preview_label.setAlignment(Qt.AlignCenter)
        self.preview_scroll_area.setWidget(self.preview_label)
        
        self.preview_button = QPushButton("Generate Preview")
        self.preview_button.clicked.connect(self.generate_preview)
        self.preview_button.setEnabled(False)
        
        preview_layout.addWidget(self.preview_scroll_area)
        preview_layout.addWidget(self.preview_button)
        preview_widget.setLayout(preview_layout)
        
        # Info panel
        info_widget = QWidget()
        info_layout = QVBoxLayout()
        
        info_header = QLabel("Image Information")
        self.info_label = QLabel("No image selected")
        
        info_layout.addWidget(info_header)
        info_layout.addWidget(self.info_label)
        
        info_widget.setLayout(info_layout)
        
        # Add preview and info to right layout
        right_layout.addWidget(preview_widget)
        right_layout.addWidget(info_widget)
        right_widget.setLayout(right_layout)
        
        # Add widgets to content splitter
        content_splitter.addWidget(left_widget)
        content_splitter.addWidget(right_widget)
        content_splitter.setSizes([300, 700])
        
        # Add to main layout
        layout.addLayout(top_layout)
        layout.addWidget(content_splitter, 1)
        
        self.setLayout(layout)
        
        self.file_list.itemClicked.connect(lambda item: self.preview_button.setEnabled(True))
        
    def apply_styles(self):
        """Apply custom styles to the UI components"""
        # Folder selection button
        self.augmentation_folder_button.setStyleSheet(AnnotationStyles.BUTTON_STYLE)
        
        self.noise_type_combo.setStyleSheet("""
                                            QComboBox {
                border: 1px solid """ + AnnotationStyles.BORDER_COLOR + """;
                border-radius: 4px;
                padding: 4px;
                background-color: white;
                min-width: 100px;
            }
            
            QComboBox::drop-down {
                subcontrol-origin: padding;
                subcontrol-position: top right;
                width: 20px;
                border-left: 1px solid """ + AnnotationStyles.BORDER_COLOR + """;
            }
            
            QComboBox::down-arrow {
                image: url(down_arrow.png);
            }
            
            QComboBox QAbstractItemView {
                border: 1px solid """ + AnnotationStyles.BORDER_COLOR + """;
                selection-background-color: """ + AnnotationStyles.SECONDARY_COLOR + """;
                selection-color: white;
            }
                                            """)
        # Dataset combo box
        self.dataset_combo.setStyleSheet("""
            QComboBox {
                border: 1px solid """ + AnnotationStyles.BORDER_COLOR + """;
                border-radius: 4px;
                padding: 4px;
                background-color: white;
                min-width: 100px;
            }
            
            QComboBox::drop-down {
                subcontrol-origin: padding;
                subcontrol-position: top right;
                width: 20px;
                border-left: 1px solid """ + AnnotationStyles.BORDER_COLOR + """;
            }
            
            QComboBox::down-arrow {
                image: url(down_arrow.png);
            }
            
            QComboBox QAbstractItemView {
                border: 1px solid """ + AnnotationStyles.BORDER_COLOR + """;
                selection-background-color: """ + AnnotationStyles.SECONDARY_COLOR + """;
                selection-color: white;
            }
        """)
        
        # Headers
        for header in self.findChildren(QLabel):
            if header.text() in ["Image Files", "Preview Augmentation", "Image Information"]:
                header.setStyleSheet(AnnotationStyles.HEADING_LABEL_STYLE)
        
        # File list
        self.file_list.setStyleSheet(AnnotationStyles.LIST_WIDGET_STYLE)
        
        # Group boxes
        for group_box in self.findChildren(QGroupBox):
            group_box.setStyleSheet(AnnotationStyles.GROUP_BOX_STYLE)
        
        # Spinboxes and DoubleSpinboxes
        spinbox_style = """
            QSpinBox, QDoubleSpinBox {
                border: 1px solid """ + AnnotationStyles.BORDER_COLOR + """;
                border-radius: 4px;
                padding: 4px;
                background: white;
            }
            
            QSpinBox::up-button, QDoubleSpinBox::up-button {
                subcontrol-origin: border;
                subcontrol-position: top right;
                width: 16px;
                border-left: 1px solid """ + AnnotationStyles.BORDER_COLOR + """;
                border-bottom: 1px solid """ + AnnotationStyles.BORDER_COLOR + """;
            }
            
            QSpinBox::down-button, QDoubleSpinBox::down-button {
                subcontrol-origin: border;
                subcontrol-position: bottom right;
                width: 16px;
                border-left: 1px solid """ + AnnotationStyles.BORDER_COLOR + """;
            }
            
            QSpinBox:focus, QDoubleSpinBox:focus {
                border: 1px solid """ + AnnotationStyles.SECONDARY_COLOR + """;
            }
        """
        
        for spinbox in self.findChildren(QSpinBox) + self.findChildren(QDoubleSpinBox):
            spinbox.setStyleSheet(spinbox_style)
        
        # Checkboxes
        checkbox_style = """
            QCheckBox {
                spacing: 5px;
            }
            
            QCheckBox::indicator {
                width: 18px;
                height: 18px;
            }
            
            QCheckBox::indicator:unchecked {
                border: 1px solid """ + AnnotationStyles.BORDER_COLOR + """;
                background-color: white;
                border-radius: 3px;
            }
            
            QCheckBox::indicator:checked {
                border: 1px solid """ + AnnotationStyles.SECONDARY_COLOR + """;
                background-color: """ + AnnotationStyles.SECONDARY_COLOR + """;
                border-radius: 3px;
                image: url(checkmark.png);
            }
        """
        
        for checkbox in self.findChildren(QCheckBox):
            checkbox.setStyleSheet(checkbox_style)
        
        # Scroll area
        self.preview_scroll_area.setStyleSheet(AnnotationStyles.SCROLL_AREA_STYLE + AnnotationStyles.SCROLLBAR_STYLE)
        
        # Preview image area
        self.preview_label.setStyleSheet("""
            QLabel {
                background-color: #F0F0F0;
                border: 1px solid """ + AnnotationStyles.BORDER_COLOR + """;
                border-radius: 4px;
            }
        """)
        
        # Buttons
        self.preview_button.setStyleSheet(AnnotationStyles.LIGHT_SECONDARY_BUTTON_STYLE)
        self.generate_button.setStyleSheet(AnnotationStyles.GRADIENT_BUTTON_STYLE)
        
        # Progress bar
        self.progress_bar.setStyleSheet("""
            QProgressBar {
                border: 1px solid """ + AnnotationStyles.BORDER_COLOR + """;
                border-radius: 4px;
                text-align: center;
                background-color: white;
            }
            
            QProgressBar::chunk {
                background-color: """ + AnnotationStyles.SUCCESS_COLOR + """;
                width: 1px;
            }
        """)
        
        # Splitter
        for splitter in self.findChildren(QSplitter):
            splitter.setStyleSheet(AnnotationStyles.SPLITTER_STYLE)
              
    def select_augmentation_folder(self):
        """Select the data folder containing train/test/valid subdirectories"""
        folder = QFileDialog.getExistingDirectory(self, "Select Data Folder")
        if folder:
            # Verify the folder has the required structure
            train_dir = os.path.join(folder, "train")
            test_dir = os.path.join(folder, "test")
            valid_dir = os.path.join(folder, "valid")
            
            required_dirs = [train_dir, test_dir, valid_dir]
            missing_dirs = []
            
            for dir_path in required_dirs:
                if not os.path.exists(dir_path):
                    missing_dirs.append(os.path.basename(dir_path))
                else:
                    # Check for subdirectories
                    images_dir = os.path.join(dir_path, "images")
                    labels_dir = os.path.join(dir_path, "labels")
                    
                    for subdir in [images_dir, labels_dir]:
                        if not os.path.exists(subdir):
                            missing_dirs.append(f"{os.path.basename(dir_path)}/{os.path.basename(subdir)}")
            
            if missing_dirs:
                QMessageBox.warning(self, "Invalid Folder Structure", 
                                   f"The selected folder is missing the following required directories:\n"
                                   f"{', '.join(missing_dirs)}")
                return
            
            self.augmentation_folder = folder
            self.selected_folder_label.setText(folder)
            self.generate_button.setEnabled(True)
            
            # Load files from the selected dataset
            self.load_dataset()
            
            if hasattr(self.parent, 'status_bar'):
                self.parent.status_bar.showMessage(f"Selected augmentation folder: {folder}")
    
    def load_dataset(self):
        """Load image files from the selected dataset (train/test/valid)"""
        if not self.augmentation_folder:
            return
        
        dataset = self.dataset_combo.currentText()
        self.file_list.clear()
        
        # Build paths
        dataset_dir = os.path.join(self.augmentation_folder, dataset)
        images_dir = os.path.join(dataset_dir, "images")
        
        if not os.path.exists(images_dir):
            self.info_label.setText(f"No images directory found in {dataset}")
            return
        
        # Find image files
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif']
        self.image_files = []
        
        for file in os.listdir(images_dir):
            file_path = os.path.join(images_dir, file)
            if os.path.isfile(file_path):
                ext = os.path.splitext(file)[1].lower()
                if ext in image_extensions:
                    self.file_list.addItem(file)
                    self.image_files.append(file)
        
        # Update info
        self.info_label.setText(f"Found {len(self.image_files)} images in {dataset}")
    
    def load_image(self, item):
        """Load the selected image and update info"""
        if not item:
            return
            
        dataset = self.dataset_combo.currentText()
        file_name = item.text()
        
        # Update current index
        self.current_image_index = self.file_list.currentRow()
        
        # Build file paths
        images_dir = os.path.join(self.augmentation_folder, dataset, "images")
        labels_dir = os.path.join(self.augmentation_folder, dataset, "labels")
        
        self.current_image_path = os.path.join(images_dir, file_name)
        self.current_image = cv2.imread(self.current_image_path)
        
        if self.current_image is None:
            self.info_label.setText(f"Failed to load image: {file_name}")
            return
        
        # Load labels if they exist
        base_name = os.path.splitext(file_name)[0]
        label_path = os.path.join(labels_dir, f"{base_name}.txt")
        
        self.current_segment = None
        self.keypoints = []
        
        # Load segmentation labels
        if os.path.exists(label_path):
            with open(label_path, 'r') as f:
                self.current_segment = f.read().strip()
                # Parse segmentation points
                parts = self.current_segment.split()
                if len(parts) > 1:
                    class_id = int(parts[0])
                    points_data = parts[1:]
                    
                    h, w = self.current_image.shape[:2]
                    
                    for i in range(0, len(points_data), 2):
                        if i + 1 < len(points_data):
                            x = float(points_data[i]) * w
                            y = float(points_data[i + 1]) * h
                            self.keypoints.append([x, y])
        
        # Update info
        h, w = self.current_image.shape[:2]
        info = f"Image: {file_name} ({w}x{h})"
        
        if self.current_segment:
            info += "\nSegmentation: Yes"
        else:
            info += "\nSegmentation: No"
            
        self.info_label.setText(info)
    
    def create_transforms(self, image_height=None, image_width=None):
        """Create albumentations transforms based on UI settings"""
        transforms = []
        
        # Add selected transforms with their probabilities
        if self.hflip_check.isChecked():
            transforms.append(A.HorizontalFlip(p=self.hflip_prob.value()))
            
        if self.rotate_check.isChecked():
            transforms.append(A.Rotate(
                limit=self.rotate_limit.value(),
                border_mode=cv2.BORDER_CONSTANT,
                value=0,
                p=self.rotate_prob.value()
            ))
            
        if self.blur_check.isChecked():
            transforms.append(A.Blur(
                blur_limit=self.blur_limit.value(),
                p=self.blur_prob.value()
            ))
        
        # Add crop transformation if enabled and image dimensions provided
        if self.crop_check.isChecked() and image_height and image_width:
            # Calculate crop dimensions
            min_height = self.crop_min_height.value() / 100.0
            min_width = self.crop_min_width.value() / 100.0
            
            crop_height = int(min_height * image_height)
            crop_width = int(min_width * image_width)
            
            transforms.append(A.RandomCrop(
                height=crop_height,
                width=crop_width,
                p=self.crop_prob.value()
            ))
            
        # Add noise transformation if enabled
        if self.noise_check.isChecked():
            noise_type = self.noise_type_combo.currentText()
            amount = self.noise_amount.value()
            
            if noise_type == "gaussian":
                transforms.append(A.GaussNoise(
                    var_limit=(10, int(amount * 255.0)),
                    p=self.noise_prob.value()
                ))
            elif noise_type == "poisson":
                transforms.append(A.MultiplicativeNoise(
                    multiplier=(1.0 - amount, 1.0 + amount),
                    elementwise=True,
                    p=self.noise_prob.value()
                ))
            elif noise_type == "salt":
                transforms.append(A.PixelDropout(
                    dropout_prob=amount,
                    per_channel=False,
                    drop_value=255,  # White pixels
                    p=self.noise_prob.value()
                ))
            elif noise_type == "pepper":
                transforms.append(A.PixelDropout(
                    dropout_prob=amount,
                    per_channel=False,
                    drop_value=0,  # Black pixels
                    p=self.noise_prob.value()
                ))
            elif noise_type == "s&p":
                # Salt and pepper noise
                transforms.append(A.PixelDropout(
                    dropout_prob=amount * 0.5,
                    per_channel=False,
                    drop_value=255,  # White pixels (salt)
                    p=self.noise_prob.value()
                ))
                transforms.append(A.PixelDropout(
                    dropout_prob=amount * 0.5,
                    per_channel=False,
                    drop_value=0,  # Black pixels (pepper)
                    p=self.noise_prob.value()
                ))
        
        return transforms
    
    def generate_augmented_data(self):
        """Generate augmented data based on current settings"""
        if not self.augmentation_folder:
            QMessageBox.warning(self, "Error", "No folder selected")
            return
            
        dataset = self.dataset_combo.currentText()
        
        # Check if we have images to process
        if not self.image_files:
            QMessageBox.warning(self, "Error", "No images to augment")
            return
            
        # Create output directory
        output_dir = os.path.join(self.augmentation_folder, f"{dataset}_augmented")
        
        # Confirm overwrite if directory exists
        if os.path.exists(output_dir):
            reply = QMessageBox.question(
                self, 
                "Confirm Overwrite",
                f"Output directory '{output_dir}' already exists. Overwrite?",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No
            )
            if reply == QMessageBox.No:
                return
        
        # Set random seed if enabled
        if self.random_seed_check.isChecked():
            random.seed(self.random_seed.value())
            np.random.seed(self.random_seed.value())
            A.random.seed(self.random_seed.value())
        
        # Start the augmentation thread
        self.augmentation_thread = AugmentationThread(
            self,
            dataset,
            self.image_files,
            output_dir,
            self.num_augmentations.value()
        )
        
        # Connect signals
        self.augmentation_thread.progress.connect(self.update_progress)
        self.augmentation_thread.completed.connect(self.augmentation_completed)
        self.augmentation_thread.error.connect(self.augmentation_error)
        
        # Disable UI during processing
        self.generate_button.setEnabled(False)
        self.progress_bar.setValue(0)
        self.progress_bar.setVisible(True)
        
        # Start thread
        self.augmentation_thread.start()
        
        
    def generate_preview(self):
        """Generate and display multiple preview versions of augmentation on the current image directly on preview_label"""
        if self.current_image is None:
            QMessageBox.warning(self, "Error", "No image selected")
            return
        
        try:
            original_image = self.current_image.copy()
            h, w = original_image.shape[:2]
            
            keypoints = []
            keypoint_labels = []
            
            if self.keypoints:
                keypoints = self.keypoints.copy()
                keypoint_labels = [0] * len(keypoints)  # Giả sử tất cả cùng class 0
            
            original_display = original_image.copy()
            if keypoints:
                for point in keypoints:
                    x, y = int(point[0]), int(point[1])
                    cv2.circle(original_display, (x, y), 5, (0, 0, 255), -1)
                
                if len(keypoints) > 2:
                    pts = np.array(keypoints, np.int32)
                    pts = pts.reshape((-1, 1, 2))
                    cv2.polylines(original_display, [pts], True, (0, 255, 0), 2)
            
            transforms = self.create_transforms(h, w)
            
            preview_pipeline = A.Compose(
                transforms, 
                keypoint_params=A.KeypointParams(format='xy', remove_invisible=False)
            )
            
            preview_images = []
            preview_images.append(("Original", original_display))
            
            for i in range(3):
                transformed = preview_pipeline(
                    image=original_image.copy(),
                    keypoints=keypoints.copy() if keypoints else [],
                )
                
                aug_image = transformed['image']
                aug_keypoints = transformed.get('keypoints', [])
                
                if aug_keypoints:
                    for point in aug_keypoints:
                        x, y = int(point[0]), int(point[1])
                        cv2.circle(aug_image, (x, y), 5, (0, 0, 255), -1)
                    
                    if len(aug_keypoints) > 2:
                        pts = np.array(aug_keypoints, np.int32)
                        pts = pts.reshape((-1, 1, 2))
                        cv2.polylines(aug_image, [pts], True, (0, 255, 0), 2)
                
                preview_images.append((f"Version {i+1}", aug_image))
            
            grid_cols = 2  
            grid_rows = (len(preview_images) + grid_cols - 1) // grid_cols
            
            thumb_height = 240  
            aspect_ratio = w / h
            thumb_width = int(thumb_height * aspect_ratio)
            
            grid_width = grid_cols * (thumb_width + 10) + 10  
            grid_height = grid_rows * (thumb_height + 40) + 10 
            
            grid_image = np.ones((grid_height, grid_width, 3), dtype=np.uint8) * 255
            
            for idx, (title, img) in enumerate(preview_images):
                row = idx // grid_cols
                col = idx % grid_cols
                
                x = col * (thumb_width + 10) + 10
                y = row * (thumb_height + 40) + 10
                
                resized_img = cv2.resize(img, (thumb_width, thumb_height))
                
                grid_image[y:y+thumb_height, x:x+thumb_width] = resized_img
                
                cv2.putText(grid_image, title, (x, y+thumb_height+20), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1, cv2.LINE_AA)
            
            grid_rgb = cv2.cvtColor(grid_image, cv2.COLOR_BGR2RGB)
            h, w = grid_rgb.shape[:2]
            bytes_per_line = 3 * w
            
            q_image = QImage(grid_rgb.data, w, h, bytes_per_line, QImage.Format_RGB888)
            pixmap = QPixmap.fromImage(q_image)
            
            self.preview_label.setPixmap(pixmap)
            self.preview_label.setFixedSize(grid_width, grid_height)
            
            self.preview_scroll_area.ensureVisible(0, 0)
            
            self.augmented_image = grid_image
            
            self.preview_button.setText("Regenerate Preview")
            
        except Exception as e:
            QMessageBox.critical(self, "Preview Error", f"Error generating preview: {str(e)}")
        
    # def generate_preview(self):
    #     """Generate and display multiple preview versions of augmentation on the current image"""
    #     if self.current_image is None:
    #         QMessageBox.warning(self, "Error", "No image selected")
    #         return
        
    #     try:
    #         original_image = self.current_image.copy()
    #         h, w = original_image.shape[:2]
            
    #         keypoints = []
    #         keypoint_labels = []
            
    #         if self.keypoints:
    #             keypoints = self.keypoints.copy()
    #             keypoint_labels = [0] * len(keypoints)  # Giả sử tất cả cùng class 0
            
    #         original_display = original_image.copy()
    #         if keypoints:
    #             for point in keypoints:
    #                 x, y = int(point[0]), int(point[1])
    #                 cv2.circle(original_display, (x, y), 5, (0, 0, 255), -1)
                
    #             if len(keypoints) > 2:
    #                 pts = np.array(keypoints, np.int32)
    #                 pts = pts.reshape((-1, 1, 2))
    #                 cv2.polylines(original_display, [pts], True, (0, 255, 0), 2)
            
    #         transforms = self.create_transforms(h, w)
            
    #         preview_pipeline = A.Compose(
    #             transforms, 
    #             keypoint_params=A.KeypointParams(format='xy', remove_invisible=False)
    #         )
            
    #         augmented_images = []
            
    #         augmented_images.append(("Original", original_display))
            
    #         for i in range(5):
    #             transformed = preview_pipeline(
    #                 image=original_image.copy(),
    #                 keypoints=keypoints.copy() if keypoints else []
                    
    #             )
                
    #             aug_image = transformed['image']
    #             aug_keypoints = transformed.get('keypoints', [])
                
    #             if aug_keypoints:
    #                 for point in aug_keypoints:
    #                     x, y = int(point[0]), int(point[1])
    #                     cv2.circle(aug_image, (x, y), 5, (0, 0, 255), -1)
                    
    #                 if len(aug_keypoints) > 2:
    #                     pts = np.array(aug_keypoints, np.int32)
    #                     pts = pts.reshape((-1, 1, 2))
    #                     cv2.polylines(aug_image, [pts], True, (0, 255, 0), 2)
                
    #             augmented_images.append((f"Version {i+1}", aug_image))
            
    #         preview_dialog = QDialog(self)
    #         preview_dialog.setWindowTitle("Augmentation Previews")
    #         preview_dialog.setMinimumSize(900, 600)
            
    #         dialog_layout = QVBoxLayout()
            
    #         grid = QGridLayout()
            
    #         num_images = len(augmented_images)
    #         cols = 3  
    #         rows = (num_images + cols - 1) // cols
            
    #         for idx, (title, img) in enumerate(augmented_images):
    #             row = idx // cols
    #             col = idx % cols
                
    #             img_widget = QWidget()
    #             img_layout = QVBoxLayout()
                
    #             title_label = QLabel(title)
    #             title_label.setAlignment(Qt.AlignCenter)
    #             img_layout.addWidget(title_label)
                
    #             img_label = QLabel()
                
    #             img_h, img_w = img.shape[:2]
    #             bytes_per_line = 3 * img_w
    #             img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    #             q_image = QImage(img_rgb.data, img_w, img_h, bytes_per_line, QImage.Format_RGB888)
                
    #             # Scale ảnh để vừa với giao diện
    #             pixmap = QPixmap.fromImage(q_image)
    #             scaled_pixmap = pixmap.scaled(250, 250, Qt.KeepAspectRatio, Qt.SmoothTransformation)
                
    #             img_label.setPixmap(scaled_pixmap)
    #             img_label.setAlignment(Qt.AlignCenter)
    #             img_layout.addWidget(img_label)
                
    #             img_widget.setLayout(img_layout)
    #             grid.addWidget(img_widget, row, col)
            
    #         dialog_layout.addLayout(grid)
            
    #         button_layout = QHBoxLayout()
            
    #         regenerate_button = QPushButton("Regenerate")
    #         regenerate_button.clicked.connect(lambda: self.regenerate_previews(preview_dialog, original_image, keypoints, keypoint_labels))
            
    #         close_button = QPushButton("Close")
    #         close_button.clicked.connect(preview_dialog.accept)
            
    #         button_layout.addWidget(regenerate_button)
    #         button_layout.addWidget(close_button)
            
    #         dialog_layout.addLayout(button_layout)
            
    #         preview_dialog.setLayout(dialog_layout)
    #         preview_dialog.exec_()
            
    #     except Exception as e:
    #         QMessageBox.critical(self, "Preview Error", f"Error generating preview: {str(e)}")

    def regenerate_previews(self, dialog, original_image, keypoints, keypoint_labels):
        """Regenerate preview versions with current settings"""
        dialog.accept()  
        self.generate_preview() 


    def augment_single_image(self, image_file, dataset, output_dir, num_augmentations):
        """Augment a single image and its labels"""
        # Create output subdirectories
        output_images_dir = os.path.join(output_dir, "images")
        output_labels_dir = os.path.join(output_dir, "labels")
        
        os.makedirs(output_images_dir, exist_ok=True)
        os.makedirs(output_labels_dir, exist_ok=True)
        
        # Source paths
        base_name = os.path.splitext(image_file)[0]
        ext = os.path.splitext(image_file)[1]
        
        source_images_dir = os.path.join(self.augmentation_folder, dataset, "images")
        source_labels_dir = os.path.join(self.augmentation_folder, dataset, "labels")
        
        # Load image
        image_path = os.path.join(source_images_dir, image_file)
        image = cv2.imread(image_path)
        if image is None:
            return
        
        h, w = image.shape[:2]
        
        # Load labels
        keypoints = []
        keypoint_labels = []
        
        # Load segmentation labels
        segment_path = os.path.join(source_labels_dir, f"{base_name}.txt")
        if os.path.exists(segment_path):
            with open(segment_path, 'r') as f:
                segment_data = f.read().strip()
                
            # Parse segmentation points
            parts = segment_data.split()
            if len(parts) > 1:
                class_id = int(parts[0])
                points_data = parts[1:]
                
                for i in range(0, len(points_data), 2):
                    if i + 1 < len(points_data):
                        x = float(points_data[i])  # Already normalized
                        y = float(points_data[i + 1])
                        keypoints.append([x * w, y * h])  # De-normalize for albumentations
                        keypoint_labels.append(class_id)
        
        # Create transforms for this image
        transforms = self.create_transforms(h, w)
        
        # Create the dynamic pipeline
        dynamic_pipeline = A.Compose(
            transforms, 
            keypoint_params=A.KeypointParams(format='xy', remove_invisible=False)
        )
        
        # Generate augmentations
        for i in range(num_augmentations):
            try:
                # Apply transformations
                transformed = dynamic_pipeline(
                    image=image,
                    keypoints=keypoints,
                )
                
                augmented_image = transformed['image']
                augmented_keypoints = transformed['keypoints']
                
                # Save augmented image
                aug_image_file = f"{base_name}_aug{i}{ext}"
                aug_image_path = os.path.join(output_images_dir, aug_image_file)
                cv2.imwrite(aug_image_path, augmented_image)
                
                # Save augmented segmentation labels
                if keypoints and augmented_keypoints:
                    aug_labels_path = os.path.join(output_labels_dir, f"{base_name}_aug{i}.txt")
                    
                    # Format segmentation in YOLO format
                    if keypoint_labels:
                        class_id = keypoint_labels[0]  # Assuming all points have same class
                    else:
                        class_id = 0
                        
                    # Normalize coordinates
                    aug_h, aug_w = augmented_image.shape[:2]
                    normalized_points = []
                    
                    for kp in augmented_keypoints:
                        x, y = kp
                        x_norm = x / aug_w
                        y_norm = y / aug_h
                        normalized_points.extend([x_norm, y_norm])
                    
                    # Format: class_id x1 y1 x2 y2 ...
                    segment_line = f"{class_id} " + " ".join([f"{p:.15f}" for p in normalized_points])
                    
                    with open(aug_labels_path, 'w') as f:
                        f.write(segment_line)
                    
            except Exception as e:
                print(f"Error augmenting {image_file} (augmentation {i}): {str(e)}")
                continue
    
    def update_progress(self, value):
        """Update progress bar"""
        self.progress_bar.setValue(value)
    
    def augmentation_completed(self, message):
        """Handle completion of augmentation"""
        self.progress_bar.setVisible(False)
        self.generate_button.setEnabled(True)
        QMessageBox.information(self, "Augmentation Complete", message)
        if hasattr(self.parent, 'status_bar'):
            self.parent.status_bar.showMessage(message)
    
    def augmentation_error(self, error_message):
        """Handle error in augmentation"""
        self.progress_bar.setVisible(False)
        self.generate_button.setEnabled(True)
        QMessageBox.critical(self, "Augmentation Error", error_message)
        if hasattr(self.parent, 'status_bar'):
            self.parent.status_bar.showMessage("Error: " + error_message)
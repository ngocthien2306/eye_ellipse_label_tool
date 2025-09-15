import os
import yaml
import tempfile
import threading
import subprocess
from datetime import datetime
from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, 
                           QFileDialog, QMessageBox, QComboBox, QSpinBox, QDoubleSpinBox,
                           QCheckBox, QGroupBox, QGridLayout, QTabWidget, QTextEdit,
                           QLineEdit, QFormLayout, QListWidget, QSplitter, QProgressBar)
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QTimer
import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas

from src.utils.styles import AnnotationStyles

class TrainingThread(QThread):
    """Thread for running Ultralytics YOLO11 training in background"""
    progress = pyqtSignal(str)
    completed = pyqtSignal(str)
    error = pyqtSignal(str)
    metrics_update = pyqtSignal(dict)
    
    def __init__(self, training_args):
        super().__init__()
        self.training_args = training_args
        self.process = None
        self.terminated = False
    
    def run(self):
        try:
            if self.training_args.get('use_custom_yaml', False):
                yaml_path = self.training_args['yaml_path']
            else:
                # Create temporary YAML for dataset config
                yaml_path = self._create_temp_yaml()
            
            # Prepare YOLO command
            cmd = [
                "yolo", "task=segment", "train",
                f"data={yaml_path}",
                f"model={self.training_args['model']}",
                f"epochs={self.training_args['epochs']}",
                f"imgsz={self.training_args['imgsz']}",
                f"batch={self.training_args['batch']}",
                f"patience={self.training_args['patience']}"
            ]
            
            # Add optional arguments
            if self.training_args.get('workers', 0) > 0:
                cmd.append(f"workers={self.training_args['workers']}")
            
            if self.training_args.get('lr0'):
                cmd.append(f"lr0={self.training_args['lr0']}")
                
            if self.training_args.get('optimizer'):
                cmd.append(f"optimizer={self.training_args['optimizer']}")
                
            if self.training_args.get('augment', False):
                cmd.append("augment=True")
                
            if self.training_args.get('project'):
                cmd.append(f"project={self.training_args['project']}")
                
            if self.training_args.get('name'):
                cmd.append(f"name={self.training_args['name']}")
                
            # Run YOLO training command
            self.process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                universal_newlines=True
            )
            
            # Metrics dictionary
            metrics = {
                'epoch': [],
                'training_loss': [],
                'validation_loss': [],
                'precision': [],
                'recall': [],
                'mAP50': [],
                'mAP50-95': []
            }
            
            # Process output line by line
            for line in iter(self.process.stdout.readline, ''):
                if self.terminated:
                    if self.process:
                        self.process.terminate()
                    break
                    
                self.progress.emit(line.strip())
                
                # Extract metrics if it's a training progress line
                if "Epoch" in line and "/box_loss" in line:
                    try:
                        parts = line.strip().split()
                        epoch_info = parts[0].split('/')
                        current_epoch = int(epoch_info[0].replace('Epoch', '').strip())
                        total_epochs = int(epoch_info[1])
                        
                        loss_idx = line.find("loss:")
                        box_idx = line.find("/box_loss:")
                        
                        if loss_idx > 0 and box_idx > 0:
                            loss_val = float(line[loss_idx+5:box_idx].strip())
                            
                            metrics['epoch'].append(current_epoch)
                            metrics['training_loss'].append(loss_val)
                            
                            # Emit updated metrics
                            self.metrics_update.emit(metrics)
                    except Exception as e:
                        print(f"Error parsing metrics: {str(e)}")
                
                # Extract validation metrics
                if "mAP50" in line and "mAP50-95" in line:
                    try:
                        parts = line.strip().split()
                        for part in parts:
                            if "val_loss:" in part:
                                val_loss = float(part.split(":")[-1])
                                if len(metrics['validation_loss']) < len(metrics['epoch']):
                                    metrics['validation_loss'].append(val_loss)
                                else:
                                    metrics['validation_loss'][-1] = val_loss
                            
                            if "precision:" in part:
                                precision = float(part.split(":")[-1])
                                if len(metrics['precision']) < len(metrics['epoch']):
                                    metrics['precision'].append(precision)
                                else:
                                    metrics['precision'][-1] = precision
                            
                            if "recall:" in part:
                                recall = float(part.split(":")[-1])
                                if len(metrics['recall']) < len(metrics['epoch']):
                                    metrics['recall'].append(recall)
                                else:
                                    metrics['recall'][-1] = recall
                            
                            if "mAP50:" in part:
                                mAP50 = float(part.split(":")[-1])
                                if len(metrics['mAP50']) < len(metrics['epoch']):
                                    metrics['mAP50'].append(mAP50)
                                else:
                                    metrics['mAP50'][-1] = mAP50
                            
                            if "mAP50-95:" in part:
                                mAP50_95 = float(part.split(":")[-1])
                                if len(metrics['mAP50-95']) < len(metrics['epoch']):
                                    metrics['mAP50-95'].append(mAP50_95)
                                else:
                                    metrics['mAP50-95'][-1] = mAP50_95
                        
                        # Emit updated metrics
                        self.metrics_update.emit(metrics)
                    except Exception as e:
                        print(f"Error parsing validation metrics: {str(e)}")
            
            # Get return code
            return_code = self.process.wait()
            
            if return_code == 0:
                result_path = os.path.join(
                    self.training_args.get('project', 'runs/detect/train'),
                    self.training_args.get('name', 'exp')
                )
                self.completed.emit(f"Training completed successfully! Results saved to: {result_path}")
            else:
                self.error.emit(f"Training process returned code {return_code}")
                
        except Exception as e:
            self.error.emit(f"Error during training: {str(e)}")
    
    def stop(self):
        """Stop the training process"""
        self.terminated = True
        if self.process:
            self.process.terminate()
    
    def _create_temp_yaml(self):
        """Create a temporary YAML file for the dataset configuration"""
        dataset_dir = self.training_args['dataset_dir']
        
        # Define dataset configuration
        dataset_config = {
            'path': dataset_dir,
            'train': 'train/images',
            'val': 'valid/images',
            'test': 'test/images',
            'names': {
                0: 'pupil'
            }
        }
        
        # Create temporary file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as yaml_file:
            yaml.dump(dataset_config, yaml_file, default_flow_style=False)
            yaml_path = yaml_file.name
        
        return yaml_path

class MetricsCanvas(FigureCanvas):
    """Canvas for plotting training metrics"""
    def __init__(self, width=8, height=6, dpi=100):
        self.fig, self.axes = plt.subplots(2, 2, figsize=(width, height), dpi=dpi)
        super(MetricsCanvas, self).__init__(self.fig)
        
        self.fig.tight_layout(pad=3.0)
        self.metrics = {
            'epoch': [],
            'training_loss': [],
            'validation_loss': [],
            'precision': [],
            'recall': [],
            'mAP50': [],
            'mAP50-95': []
        }
    
    def update_metrics(self, metrics):
        """Update the metrics and redraw plots"""
        self.metrics = metrics
        self.redraw()
    
    def redraw(self):
        """Redraw all plots with current metrics"""
        # Clear all axes
        for ax_row in self.axes:
            for ax in ax_row:
                ax.clear()
        
        # Plot training and validation loss
        ax_loss = self.axes[0, 0]
        if self.metrics['epoch'] and self.metrics['training_loss']:
            ax_loss.plot(self.metrics['epoch'], self.metrics['training_loss'], 'b-', label='Training Loss')
            
            if self.metrics['validation_loss']:
                # Make sure validation loss array is the same length as epoch
                val_loss = self.metrics['validation_loss']
                if len(val_loss) < len(self.metrics['epoch']):
                    val_loss = val_loss + [None] * (len(self.metrics['epoch']) - len(val_loss))
                ax_loss.plot(self.metrics['epoch'], val_loss, 'r-', label='Validation Loss')
                
            ax_loss.set_xlabel('Epoch')
            ax_loss.set_ylabel('Loss')
            ax_loss.set_title('Training/Validation Loss')
            ax_loss.legend()
            ax_loss.grid(True, linestyle='--', alpha=0.7)
        
        # Plot precision and recall
        ax_pr = self.axes[0, 1]
        if self.metrics['epoch'] and self.metrics['precision'] and self.metrics['recall']:
            # Make sure arrays are the same length as epoch
            precision = self.metrics['precision']
            recall = self.metrics['recall']
            
            if len(precision) < len(self.metrics['epoch']):
                precision = precision + [None] * (len(self.metrics['epoch']) - len(precision))
            if len(recall) < len(self.metrics['epoch']):
                recall = recall + [None] * (len(self.metrics['epoch']) - len(recall))
                
            ax_pr.plot(self.metrics['epoch'], precision, 'g-', label='Precision')
            ax_pr.plot(self.metrics['epoch'], recall, 'm-', label='Recall')
            ax_pr.set_xlabel('Epoch')
            ax_pr.set_ylabel('Value')
            ax_pr.set_title('Precision/Recall')
            ax_pr.legend()
            ax_pr.grid(True, linestyle='--', alpha=0.7)
        
        # Plot mAP@0.5
        ax_map50 = self.axes[1, 0]
        if self.metrics['epoch'] and self.metrics['mAP50']:
            # Make sure array is the same length as epoch
            mAP50 = self.metrics['mAP50']
            if len(mAP50) < len(self.metrics['epoch']):
                mAP50 = mAP50 + [None] * (len(self.metrics['epoch']) - len(mAP50))
                
            ax_map50.plot(self.metrics['epoch'], mAP50, 'c-')
            ax_map50.set_xlabel('Epoch')
            ax_map50.set_ylabel('mAP@0.5')
            ax_map50.set_title('mAP@0.5')
            ax_map50.grid(True, linestyle='--', alpha=0.7)
        
        # Plot mAP@0.5:0.95
        ax_map50_95 = self.axes[1, 1]
        if self.metrics['epoch'] and self.metrics['mAP50-95']:
            # Make sure array is the same length as epoch
            mAP50_95 = self.metrics['mAP50-95']
            if len(mAP50_95) < len(self.metrics['epoch']):
                mAP50_95 = mAP50_95 + [None] * (len(self.metrics['epoch']) - len(mAP50_95))
                
            ax_map50_95.plot(self.metrics['epoch'], mAP50_95, 'y-')
            ax_map50_95.set_xlabel('Epoch')
            ax_map50_95.set_ylabel('mAP@0.5:0.95')
            ax_map50_95.set_title('mAP@0.5:0.95')
            ax_map50_95.grid(True, linestyle='--', alpha=0.7)
        
        # Update layout and draw
        self.fig.tight_layout(pad=3.0)
        self.draw()

class TrainingTab(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.parent = parent
        
        # Initialize properties
        self.dataset_dir = None
        self.yaml_path = None
        self.training_thread = None
        self.metrics = {}
        
        # Setup the UI
        self.setup_ui()
        self.apply_styles()
    
    def setup_ui(self):
        layout = QVBoxLayout()
        
        # Top section - dataset selection
        top_layout = QHBoxLayout()
        self.dataset_dir_button = QPushButton("Select Dataset Folder")
        self.dataset_dir_button.clicked.connect(self.select_dataset_dir)
        
        self.selected_dir_label = QLabel("No dataset selected")
        
        self.yaml_config_button = QPushButton("Custom YAML Config")
        self.yaml_config_button.clicked.connect(self.select_yaml_config)
        
        self.yaml_config_check = QCheckBox("Use Custom YAML")
        self.yaml_config_check.setChecked(False)
        
        top_layout.addWidget(self.dataset_dir_button)
        top_layout.addWidget(self.selected_dir_label, 1)
        top_layout.addWidget(self.yaml_config_check)
        top_layout.addWidget(self.yaml_config_button)
        
        # Main content with settings and results
        content_splitter = QSplitter(Qt.Horizontal)
        
        # Left panel - training settings
        left_widget = QWidget()
        left_layout = QVBoxLayout()
        
        # Model settings
        model_group = QGroupBox("Model Settings")
        model_layout = QFormLayout()
        
        # Model selection
        self.model_combo = QComboBox()
        self.model_combo.addItems([
            "yolo11n-seg.pt", 
            "yolo11s-seg.pt", 
            "yolo11m-seg.pt", 
            "yolo11l-seg.pt", 
            "yolo11x-seg.pt",
            "Custom Weights..."
        ])
        self.model_combo.currentIndexChanged.connect(self.handle_model_selection)
        
        self.custom_weights_path = QLineEdit()
        self.custom_weights_path.setReadOnly(True)
        self.custom_weights_path.setVisible(False)
        
        self.select_weights_button = QPushButton("Select Weights")
        self.select_weights_button.clicked.connect(self.select_custom_weights)
        self.select_weights_button.setVisible(False)
        
        # Image size
        self.imgsz_spin = QSpinBox()
        self.imgsz_spin.setRange(32, 1280)
        self.imgsz_spin.setSingleStep(32)
        self.imgsz_spin.setValue(640)
        
        # Batch size
        self.batch_spin = QSpinBox()
        self.batch_spin.setRange(1, 64)
        self.batch_spin.setValue(16)
        
        # Epochs
        self.epochs_spin = QSpinBox()
        self.epochs_spin.setRange(1, 500)
        self.epochs_spin.setValue(100)
        
        # Patience for early stopping
        self.patience_spin = QSpinBox()
        self.patience_spin.setRange(0, 100)
        self.patience_spin.setValue(30)
        
        # Workers
        self.workers_spin = QSpinBox()
        self.workers_spin.setRange(0, 16)
        self.workers_spin.setValue(8)
        
        # Add to model layout
        model_layout.addRow(QLabel("Model:"), self.model_combo)
        model_layout.addRow("", self.custom_weights_path)
        model_layout.addRow("", self.select_weights_button)
        model_layout.addRow(QLabel("Image Size:"), self.imgsz_spin)
        model_layout.addRow(QLabel("Batch Size:"), self.batch_spin)
        model_layout.addRow(QLabel("Epochs:"), self.epochs_spin)
        model_layout.addRow(QLabel("Patience:"), self.patience_spin)
        model_layout.addRow(QLabel("Workers:"), self.workers_spin)
        
        model_group.setLayout(model_layout)
        
        # Optimization settings
        optim_group = QGroupBox("Optimization Settings")
        optim_layout = QFormLayout()
        
        # Learning rate
        self.lr_spin = QDoubleSpinBox()
        self.lr_spin.setRange(0.0001, 0.1)
        self.lr_spin.setSingleStep(0.0001)
        self.lr_spin.setDecimals(6)
        self.lr_spin.setValue(0.01)
        
        # Optimizer selection
        self.optimizer_combo = QComboBox()
        self.optimizer_combo.addItems(["SGD", "Adam", "AdamW"])
        self.optimizer_combo.setCurrentText("SGD")
        
        # Augmentation
        self.augment_check = QCheckBox("Enable Augmentation")
        self.augment_check.setChecked(True)
        
        # Add to optimization layout
        optim_layout.addRow(QLabel("Learning Rate:"), self.lr_spin)
        optim_layout.addRow(QLabel("Optimizer:"), self.optimizer_combo)
        optim_layout.addRow(QLabel("Augmentation:"), self.augment_check)
        
        optim_group.setLayout(optim_layout)
        
        # Output settings
        output_group = QGroupBox("Output Settings")
        output_layout = QFormLayout()
        
        # Project name
        self.project_edit = QLineEdit("runs/segment")
        
        # Experiment name
        self.name_edit = QLineEdit("train")
        
        # Add to output layout
        output_layout.addRow(QLabel("Project:"), self.project_edit)
        output_layout.addRow(QLabel("Name:"), self.name_edit)
        
        output_group.setLayout(output_layout)
        
        # Control buttons
        buttons_layout = QHBoxLayout()
        
        self.train_button = QPushButton("Start Training")
        self.train_button.setEnabled(False)
        self.train_button.clicked.connect(self.start_training)
        
        self.stop_button = QPushButton("Stop Training")
        self.stop_button.setEnabled(False)
        self.stop_button.clicked.connect(self.stop_training)
        
        buttons_layout.addWidget(self.train_button)
        buttons_layout.addWidget(self.stop_button)
        
        # Add components to left layout
        left_layout.addWidget(model_group)
        left_layout.addWidget(optim_group)
        left_layout.addWidget(output_group)
        left_layout.addLayout(buttons_layout)
        left_layout.addStretch()
        left_widget.setLayout(left_layout)
        
        # Right panel - training results and logs
        right_widget = QTabWidget()
        
        # Metrics tab
        metrics_widget = QWidget()
        metrics_layout = QVBoxLayout()
        
        # Create metrics canvas
        self.metrics_canvas = MetricsCanvas()
        
        metrics_layout.addWidget(self.metrics_canvas)
        metrics_widget.setLayout(metrics_layout)
        
        # Logs tab
        logs_widget = QWidget()
        logs_layout = QVBoxLayout()
        
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        
        logs_layout.addWidget(self.log_text)
        logs_widget.setLayout(logs_layout)
        
        # Add tabs
        right_widget.addTab(metrics_widget, "Training Metrics")
        right_widget.addTab(logs_widget, "Training Logs")
        
        # Add widgets to content splitter
        content_splitter.addWidget(left_widget)
        content_splitter.addWidget(right_widget)
        content_splitter.setSizes([300, 700])
        
        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setTextVisible(False)
        self.progress_bar.setVisible(False)
        
        # Add to main layout
        layout.addLayout(top_layout)
        layout.addWidget(content_splitter, 1)
        layout.addWidget(self.progress_bar)
        
        self.setLayout(layout)
    
    def apply_styles(self):
        """Apply custom styles to the UI components"""
        # Buttons
        self.dataset_dir_button.setStyleSheet(AnnotationStyles.BUTTON_STYLE)
        self.yaml_config_button.setStyleSheet(AnnotationStyles.LIGHT_SECONDARY_BUTTON_STYLE)
        self.train_button.setStyleSheet(AnnotationStyles.SUCCESS_BUTTON_STYLE)
        self.stop_button.setStyleSheet(AnnotationStyles.DANGER_BUTTON_STYLE)
        self.select_weights_button.setStyleSheet(AnnotationStyles.LIGHT_SECONDARY_BUTTON_STYLE)
        
        # Group boxes
        for group_box in self.findChildren(QGroupBox):
            group_box.setStyleSheet(AnnotationStyles.GROUP_BOX_STYLE)
        
        # Combo boxes
        combobox_style = """
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
            
            QComboBox QAbstractItemView {
                border: 1px solid """ + AnnotationStyles.BORDER_COLOR + """;
                selection-background-color: """ + AnnotationStyles.SECONDARY_COLOR + """;
                selection-color: white;
            }
        """
        
        for combo in [self.model_combo, self.optimizer_combo]:
            combo.setStyleSheet(combobox_style)
        
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
        
        # Line edits
        line_edit_style = """
            QLineEdit {
                border: 1px solid """ + AnnotationStyles.BORDER_COLOR + """;
                border-radius: 4px;
                padding: 4px;
                background-color: white;
            }
            
            QLineEdit:focus {
                border: 1px solid """ + AnnotationStyles.SECONDARY_COLOR + """;
            }
            
            QLineEdit:disabled {
                background-color: #F5F5F5;
            }
        """
        
        for line_edit in [self.custom_weights_path, self.project_edit, self.name_edit]:
            line_edit.setStyleSheet(line_edit_style)
        
        # CheckBoxes
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
            }
        """
        
        for checkbox in [self.yaml_config_check, self.augment_check]:
            checkbox.setStyleSheet(checkbox_style)
        
        # Text edit
        self.log_text.setStyleSheet(AnnotationStyles.TEXT_EDIT_STYLE)
        
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
                width: 10px;
                margin: 0.5px;
            }
        """)
        
        # Splitter
        for splitter in self.findChildren(QSplitter):
            splitter.setStyleSheet(AnnotationStyles.SPLITTER_STYLE)
    
    def handle_model_selection(self, index):
        """Handle model selection changes"""
        if self.model_combo.currentText() == "Custom Weights...":
            self.custom_weights_path.setVisible(True)
            self.select_weights_button.setVisible(True)
        else:
            self.custom_weights_path.setVisible(False)
            self.select_weights_button.setVisible(False)
    
    def select_custom_weights(self):
        """Open file dialog to select custom model weights"""
        weights_file, _ = QFileDialog.getOpenFileName(
            self, "Select Model Weights", "", "PyTorch Models (*.pt);;All Files (*)"
        )
        
        if weights_file:
            self.custom_weights_path.setText(weights_file)
    
    def select_dataset_dir(self):
        """Select dataset directory containing train/valid/test folders"""
        folder = QFileDialog.getExistingDirectory(self, "Select Dataset Folder")
        if folder:
            # Verify the folder has the required structure
            train_dir = os.path.join(folder, "train")
            valid_dir = os.path.join(folder, "valid")
            test_dir = os.path.join(folder, "test")
            
            required_dirs = [train_dir, valid_dir, test_dir]
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
                QMessageBox.warning(self, "Invalid Dataset Structure", 
                                   f"The selected folder is missing the following required directories:\n"
                                   f"{', '.join(missing_dirs)}")
                return
            
            self.dataset_dir = folder
            self.selected_dir_label.setText(folder)
            self.train_button.setEnabled(True)
            
            if hasattr(self.parent, 'status_bar'):
                self.parent.status_bar.showMessage(f"Selected dataset folder: {folder}")
    
    def select_yaml_config(self):
        """Select custom YAML configuration file"""
        yaml_file, _ = QFileDialog.getOpenFileName(
            self, "Select YAML Configuration", "", "YAML Files (*.yaml *.yml);;All Files (*)"
        )
        
        if yaml_file:
            self.yaml_path = yaml_file
            # Enable checkbox to use this config
            self.yaml_config_check.setChecked(True)
            
            if hasattr(self.parent, 'status_bar'):
                self.parent.status_bar.showMessage(f"Selected YAML configuration: {yaml_file}")
    
    def start_training(self):
        """Start the training process using selected settings"""
        if not self.dataset_dir and not (self.yaml_config_check.isChecked() and self.yaml_path):
            QMessageBox.warning(self, "Warning", "Please select a dataset folder or a custom YAML configuration")
            return
        
        # Get model selection
        if self.model_combo.currentText() == "Custom Weights..." and not self.custom_weights_path.text():
            QMessageBox.warning(self, "Warning", "Please select custom weights file")
            return
        
        model = self.model_combo.currentText()
        if model == "Custom Weights...":
            model = self.custom_weights_path.text()
        
        # Prepare training arguments
        training_args = {
            'dataset_dir': self.dataset_dir,
            'model': model,
            'imgsz': self.imgsz_spin.value(),
            'batch': self.batch_spin.value(),
            'epochs': self.epochs_spin.value(),
            'patience': self.patience_spin.value(),
            'workers': self.workers_spin.value(),
            'lr0': self.lr_spin.value(),
            'optimizer': self.optimizer_combo.currentText(),
            'augment': self.augment_check.isChecked(),
            'project': self.project_edit.text(),
            'name': self.name_edit.text(),
            'use_custom_yaml': self.yaml_config_check.isChecked(),
            'yaml_path': self.yaml_path if self.yaml_config_check.isChecked() else None
        }
        
        # Create and start training thread
        self.training_thread = TrainingThread(training_args)
        self.training_thread.progress.connect(self.update_training_progress)
        self.training_thread.completed.connect(self.training_completed)
        self.training_thread.error.connect(self.training_error)
        self.training_thread.metrics_update.connect(self.update_metrics)
        
        # Update UI state
        self.train_button.setEnabled(False)
        self.stop_button.setEnabled(True)
        self.progress_bar.setVisible(True)
        
        # Start indeterminate progress bar animation
        self.progress_bar.setRange(0, 0)
        
        # Clear previous logs
        self.log_text.clear()
        
        # Start thread
        self.training_thread.start()
        
        # Log start
        start_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.log_text.append(f"=== Training started at {start_time} ===")
        self.log_text.append(f"Model: {model}")
        self.log_text.append(f"Image Size: {self.imgsz_spin.value()}")
        self.log_text.append(f"Batch Size: {self.batch_spin.value()}")
        self.log_text.append(f"Epochs: {self.epochs_spin.value()}")
        self.log_text.append(f"Optimizer: {self.optimizer_combo.currentText()}")
        self.log_text.append(f"Learning Rate: {self.lr_spin.value()}")
        self.log_text.append(f"Augmentation: {'Enabled' if self.augment_check.isChecked() else 'Disabled'}")
        self.log_text.append("=" * 40)
        
        if hasattr(self.parent, 'status_bar'):
            self.parent.status_bar.showMessage("Training started...")
    
    def stop_training(self):
        """Stop the training process"""
        if self.training_thread and self.training_thread.isRunning():
            reply = QMessageBox.question(
                self, 
                "Confirm Stop", 
                "Are you sure you want to stop the training? Progress will be lost.",
                QMessageBox.Yes | QMessageBox.No, 
                QMessageBox.No
            )
            
            if reply == QMessageBox.Yes:
                self.log_text.append("\n=== Training interrupted by user ===")
                self.training_thread.stop()
                
                # Reset UI state
                self.train_button.setEnabled(True)
                self.stop_button.setEnabled(False)
                self.progress_bar.setVisible(False)
                
                if hasattr(self.parent, 'status_bar'):
                    self.parent.status_bar.showMessage("Training stopped by user")
    
    def update_training_progress(self, text):
        """Update the log with training progress"""
        if text.strip():
            self.log_text.append(text)
            # Auto-scroll to bottom
            self.log_text.verticalScrollBar().setValue(self.log_text.verticalScrollBar().maximum())
    
    def update_metrics(self, metrics):
        """Update the metrics chart with new data"""
        self.metrics_canvas.update_metrics(metrics)
    
    def training_completed(self, message):
        """Handle training completion"""
        self.log_text.append("\n" + message)
        
        # Reset UI state
        self.train_button.setEnabled(True)
        self.stop_button.setEnabled(False)
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(100)
        
        # Show message
        QMessageBox.information(self, "Training Complete", message)
        
        if hasattr(self.parent, 'status_bar'):
            self.parent.status_bar.showMessage("Training completed")
    
    def training_error(self, error_message):
        """Handle training error"""
        self.log_text.append("\nERROR: " + error_message)
        
        # Reset UI state
        self.train_button.setEnabled(True)
        self.stop_button.setEnabled(False)
        self.progress_bar.setVisible(False)
        
        # Show error message
        QMessageBox.critical(self, "Training Error", error_message)
        
        if hasattr(self.parent, 'status_bar'):
            self.parent.status_bar.showMessage("Training error occurred")
import sys
import os
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QTabWidget, 
                             QStatusBar, QMessageBox)
from PyQt5.QtGui import QKeySequence
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QShortcut

from annotation import AnnotationTab
from split import SplitTab
from augmentation import AugmentationTab
from styles import AnnotationStyles  # Import our new styles

class EllipseDetectorApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Ellipse Detector")
        self.setGeometry(100, 100, 1300, 900)
        
        self.current_folder = ""
        self.zoom_factor = 1.0
        self.max_zoom = 5.0
        self.min_zoom = 0.5
        
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        
        self.setup_ui()
        self.setup_shortcuts()
        
        # Apply global styles to the application
        self.apply_styles()
    
    def setup_ui(self):
        main_widget = QWidget()
        main_layout = QVBoxLayout()
        
        # Tab widget for different screens
        self.tab_widget = QTabWidget()
        
        # Create tabs
        self.annotation_tab = AnnotationTab(self)
        self.split_tab = SplitTab(self)
        self.augmentation_tab = AugmentationTab(self)
        
        self.tab_widget.addTab(self.annotation_tab, "Annotation")
        self.tab_widget.addTab(self.split_tab, "Split Data")
        self.tab_widget.addTab(self.augmentation_tab, "Augmentation")
        
        main_layout.addWidget(self.tab_widget)
        main_widget.setLayout(main_layout)
        self.setCentralWidget(main_widget)
    
    def apply_styles(self):
        """Apply the custom styles to the application"""
        # Apply the status bar style
        self.status_bar.setStyleSheet("""
            QStatusBar {
                background-color: """ + AnnotationStyles.LIGHT_GRAY + """;
                border-top: 1px solid """ + AnnotationStyles.BORDER_COLOR + """;
                color: """ + AnnotationStyles.DARK_GRAY + """;
                padding: 4px;
            }
        """)
        
        # Apply tab widget style
        self.tab_widget.setStyleSheet(AnnotationStyles.TAB_STYLE)
        
        # Call the static method to apply all styles to the QApplication instance
        # This allows styles to cascade to all child widgets
        AnnotationStyles.apply_styles(QApplication.instance())
    
    def setup_shortcuts(self):
        # Zoom shortcuts
        self.shortcut_zoom_in = QShortcut(QKeySequence("Ctrl++"), self)
        self.shortcut_zoom_in.activated.connect(self.zoom_in)
        
        self.shortcut_zoom_out = QShortcut(QKeySequence("Ctrl+-"), self)
        self.shortcut_zoom_out.activated.connect(self.zoom_out)
        
        self.shortcut_zoom_reset = QShortcut(QKeySequence("Ctrl+0"), self)
        self.shortcut_zoom_reset.activated.connect(self.reset_zoom)
        
        # Undo/Redo shortcuts
        self.shortcut_undo = QShortcut(QKeySequence("Ctrl+Z"), self)
        self.shortcut_undo.activated.connect(self.prev_point)
        
        self.shortcut_redo = QShortcut(QKeySequence("Ctrl+Y"), self)
        self.shortcut_redo.activated.connect(self.next_point)
        
        # File navigation
        self.shortcut_prev_file = QShortcut(QKeySequence("Alt+Left"), self)
        self.shortcut_prev_file.activated.connect(self.prev_file)
        
        self.shortcut_next_file = QShortcut(QKeySequence("Alt+Right"), self)
        self.shortcut_next_file.activated.connect(self.next_file)
        
        # Action shortcuts
        self.shortcut_clear = QShortcut(QKeySequence("Ctrl+Delete"), self)
        self.shortcut_clear.activated.connect(self.clear_points)
        
        self.shortcut_fit = QShortcut(QKeySequence("Ctrl+F"), self)
        self.shortcut_fit.activated.connect(self.fit_ellipse)
        
        self.shortcut_save = QShortcut(QKeySequence("Ctrl+S"), self)
        self.shortcut_save.activated.connect(self.save_yolo_labels)
        
        # Process shortcut
        self.shortcut_process = QShortcut(QKeySequence("Ctrl+P"), self)
        self.shortcut_process.activated.connect(self.process_labeled_images)
    
    # Delegate to current active tab
    def zoom_in(self):
        current_tab = self.tab_widget.currentWidget()
        if hasattr(current_tab, 'zoom_in'):
            current_tab.zoom_in()
    
    def zoom_out(self):
        current_tab = self.tab_widget.currentWidget()
        if hasattr(current_tab, 'zoom_out'):
            current_tab.zoom_out()
    
    def reset_zoom(self):
        current_tab = self.tab_widget.currentWidget()
        if hasattr(current_tab, 'reset_zoom'):
            current_tab.reset_zoom()
    
    def prev_point(self):
        if hasattr(self.annotation_tab, 'prev_point'):
            self.annotation_tab.prev_point()
    
    def next_point(self):
        if hasattr(self.annotation_tab, 'next_point'):
            self.annotation_tab.next_point()
    
    def prev_file(self):
        if hasattr(self.annotation_tab, 'prev_file'):
            self.annotation_tab.prev_file()
    
    def next_file(self):
        if hasattr(self.annotation_tab, 'next_file'):
            self.annotation_tab.next_file()
    
    def clear_points(self):
        if hasattr(self.annotation_tab, 'clear_points'):
            self.annotation_tab.clear_points()
    
    def fit_ellipse(self):
        if hasattr(self.annotation_tab, 'fit_ellipse'):
            self.annotation_tab.fit_ellipse()
    
    def save_yolo_labels(self):
        if hasattr(self.annotation_tab, 'save_yolo_labels'):
            self.annotation_tab.save_yolo_labels()
    
    def process_labeled_images(self):
        if hasattr(self.annotation_tab, 'process_labeled_images'):
            self.annotation_tab.process_labeled_images()
    
    def update_info_text(self, text, append=False):
        current_tab = self.tab_widget.currentWidget()
        if hasattr(current_tab, 'update_info_text'):
            current_tab.update_info_text(text, append)
        self.status_bar.showMessage(text.split('\n')[0])

if __name__ == "__main__":
    app = QApplication(sys.argv)
    
    # You can set application-wide icon here if needed
    # app.setWindowIcon(QIcon("path/to/icon.png"))
    
    window = EllipseDetectorApp()
    window.show()
    sys.exit(app.exec_())
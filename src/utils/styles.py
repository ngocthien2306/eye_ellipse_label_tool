class AnnotationStyles:
    # Color palette
    PRIMARY_COLOR = "#3F51B5"      # Indigo
    SECONDARY_COLOR = "#2196F3"    # Blue
    SUCCESS_COLOR = "#4CAF50"      # Green
    WARNING_COLOR = "#FFC107"      # Amber
    DANGER_COLOR = "#F44336"       # Red
    INFO_COLOR = "#03A9F4"         # Light Blue
    LIGHT_GRAY = "#F5F5F5"         # Light gray for backgrounds
    DARK_GRAY = "#424242"          # Dark gray for text
    BORDER_COLOR = "#E0E0E0"       # Border color
    HIGHLIGHT_COLOR = "#FF9800"    # Orange for highlights
    
    # Button sizes
    BUTTON_HEIGHT = "28px"
    BUTTON_PADDING = "6px 12px"
    
    # Font settings
    DEFAULT_FONT = "font-family: 'Segoe UI', Arial, sans-serif;"
    BOLD_FONT = "font-weight: bold;"
    
    # Base application style
    APP_STYLE = f"""
        QWidget {{
            {DEFAULT_FONT}
            color: {DARK_GRAY};
            background-color: white;
        }}
        
        QMainWindow {{
            background-color: {LIGHT_GRAY};
        }}
        
        QStatusBar {{
            background-color: {LIGHT_GRAY};
            border-top: 1px solid {BORDER_COLOR};
            padding: 4px;
            {DEFAULT_FONT}
        }}
    """
    
    # TabWidget style
    TAB_STYLE = f"""
        QTabWidget::pane {{
            border: 1px solid {BORDER_COLOR};
            border-radius: 4px;
            background-color: white;
            padding: 4px;
        }}
        
        QTabWidget::tab-bar {{
            left: 5px;
        }}
        
        QTabBar::tab {{
            background-color: {LIGHT_GRAY};
            border: 1px solid {BORDER_COLOR};
            border-bottom-color: {BORDER_COLOR};
            border-top-left-radius: 4px;
            border-top-right-radius: 4px;
            padding: 6px 12px;
            margin-right: 2px;
            {DEFAULT_FONT}
        }}
        
        QTabBar::tab:selected {{
            background-color: white;
            border-bottom-color: white;
            font-weight: bold;
            color: {PRIMARY_COLOR};
        }}
        
        QTabBar::tab:hover:!selected {{
            background-color: #E3F2FD;
        }}
    """
    
    # Button styles
    BUTTON_STYLE = f"""
        QPushButton {{
            background-color: {SECONDARY_COLOR};
            color: white;
            border: none;
            border-radius: 4px;
            padding: {BUTTON_PADDING};
            height: {BUTTON_HEIGHT};
            {BOLD_FONT}
        }}
        
        QPushButton:hover {{
            background-color: #1976D2;
        }}
        
        QPushButton:pressed {{
            background-color: #0D47A1;
        }}
        
        QPushButton:disabled {{
            background-color: #BDBDBD;
            color: #757575;
        }}
    """
    
    # Action button variants
    SUCCESS_BUTTON_STYLE = f"""
        QPushButton {{
            background-color: {SUCCESS_COLOR};
            color: white;
            border: none;
            border-radius: 4px;
            padding: {BUTTON_PADDING};
            height: {BUTTON_HEIGHT};
            {BOLD_FONT}
        }}
        
        QPushButton:hover {{
            background-color: #388E3C;
        }}
        
        QPushButton:pressed {{
            background-color: #2E7D32;
        }}
        
        QPushButton:disabled {{
            background-color: #BDBDBD;
            color: #757575;
        }}
    """
    
    DANGER_BUTTON_STYLE = f"""
        QPushButton {{
            background-color: {DANGER_COLOR};
            color: white;
            border: none;
            border-radius: 4px;
            padding: {BUTTON_PADDING};
            height: {BUTTON_HEIGHT};
            {BOLD_FONT}
        }}
        
        QPushButton:hover {{
            background-color: #D32F2F;
        }}
        
        QPushButton:pressed {{
            background-color: #B71C1C;
        }}
        
        QPushButton:disabled {{
            background-color: #BDBDBD;
            color: #757575;
        }}
    """
    
    
    
    # Nút viền nhẹ (Outline Button)
    OUTLINE_BUTTON_STYLE = f"""
        QPushButton {{
            background-color: white;
            color: {PRIMARY_COLOR};
            border: 1px solid {PRIMARY_COLOR};
            border-radius: 4px;
            padding: {BUTTON_PADDING};
            height: {BUTTON_HEIGHT};
            {BOLD_FONT}
        }}
        
        QPushButton:hover {{
            background-color: #E8EAF6;
        }}
        
        QPushButton:pressed {{
            background-color: #C5CAE9;
        }}
        
        QPushButton:disabled {{
            background-color: white;
            border-color: #BDBDBD;
            color: #BDBDBD;
        }}
    """
    
    # Nút viền nhấn mạnh (Accent Outline Button)
    ACCENT_OUTLINE_BUTTON_STYLE = f"""
        QPushButton {{
            background-color: white;
            color: {HIGHLIGHT_COLOR};
            border: 1px solid {HIGHLIGHT_COLOR};
            border-radius: 4px;
            padding: {BUTTON_PADDING};
            height: {BUTTON_HEIGHT};
            {BOLD_FONT}
        }}
        
        QPushButton:hover {{
            background-color: #FFF3E0;
        }}
        
        QPushButton:pressed {{
            background-color: #FFE0B2;
        }}
        
        QPushButton:disabled {{
            background-color: white;
            border-color: #BDBDBD;
            color: #BDBDBD;
        }}
    """
    
    # Nút phẳng (Flat Button) - không có viền, chỉ có văn bản và màu nền khi hover
    FLAT_BUTTON_STYLE = f"""
        QPushButton {{
            background-color: transparent;
            color: {SECONDARY_COLOR};
            border: none;
            padding: {BUTTON_PADDING};
            height: {BUTTON_HEIGHT};
            {BOLD_FONT}
            text-align: left;
        }}
        
        QPushButton:hover {{
            background-color: {LIGHT_GRAY};
            border-radius: 4px;
        }}
        
        QPushButton:pressed {{
            background-color: #E0E0E0;
        }}
        
        QPushButton:disabled {{
            color: #BDBDBD;
        }}
    """
    
    # Nút có icon nhỏ (Icon Button)
    ICON_BUTTON_STYLE = f"""
        QPushButton {{
            background-color: transparent;
            border: none;
            border-radius: 16px;
            min-width: 32px;
            max-width: 32px;
            min-height: 32px;
            max-height: 32px;
        }}
        
        QPushButton:hover {{
            background-color: {LIGHT_GRAY};
        }}
        
        QPushButton:pressed {{
            background-color: #E0E0E0;
        }}
        
        QPushButton:disabled {{
            opacity: 0.5;
        }}
    """
    
    # Nút với góc tròn lớn (Pill Button)
    PILL_BUTTON_STYLE = f"""
        QPushButton {{
            background-color: {PRIMARY_COLOR};
            color: white;
            border: none;
            border-radius: 20px;
            padding: 8px 20px;
            min-height: 40px;
            {BOLD_FONT}
        }}
        
        QPushButton:hover {{
            background-color: #303F9F;
        }}
        
        QPushButton:pressed {{
            background-color: #1A237E;
        }}
        
        QPushButton:disabled {{
            background-color: #BDBDBD;
            color: #757575;
        }}
    """
    
    # Nút gradient (Gradient Button)
    GRADIENT_BUTTON_STYLE = f"""
        QPushButton {{
            background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 {PRIMARY_COLOR}, stop:1 {SECONDARY_COLOR});
            color: white;
            border: none;
            border-radius: 4px;
            padding: {BUTTON_PADDING};
            height: {BUTTON_HEIGHT};
            {BOLD_FONT}
        }}
        
        QPushButton:hover {{
            background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #303F9F, stop:1 #1976D2);
        }}
        
        QPushButton:pressed {{
            background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #1A237E, stop:1 #0D47A1);
        }}
        
        QPushButton:disabled {{
            background: #BDBDBD;
            color: #757575;
        }}
    """
    
    # Nút có hiệu ứng nổi (Elevated Button)
    ELEVATED_BUTTON_STYLE = f"""
        QPushButton {{
            background-color: white;
            color: {DARK_GRAY};
            border: none;
            border-radius: 4px;
            padding: {BUTTON_PADDING};
            height: {BUTTON_HEIGHT};
            {BOLD_FONT}
            box-shadow: 0 2px 4px rgba(0, 0, 0, 0.2);
        }}
        
        QPushButton:hover {{
            background-color: #F5F5F5;
            box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2);
        }}
        
        QPushButton:pressed {{
            background-color: #EEEEEE;
            box-shadow: 0 1px 2px rgba(0, 0, 0, 0.2);
        }}
        
        QPushButton:disabled {{
            background-color: #F5F5F5;
            color: #BDBDBD;
            box-shadow: none;
        }}
    """
    
    # Nút đã chọn (Toggle Button)
    TOGGLE_BUTTON_STYLE = f"""
        QPushButton {{
            background-color: white;
            color: {DARK_GRAY};
            border: 1px solid {BORDER_COLOR};
            border-radius: 4px;
            padding: {BUTTON_PADDING};
            height: {BUTTON_HEIGHT};
        }}
        
        QPushButton:hover {{
            background-color: {LIGHT_GRAY};
        }}
        
        QPushButton:checked {{
            background-color: {PRIMARY_COLOR};
            color: white;
            border-color: {PRIMARY_COLOR};
        }}
        
        QPushButton:disabled {{
            background-color: #F5F5F5;
            color: #BDBDBD;
            border-color: {BORDER_COLOR};
        }}
    """
    
    # Nút cảnh báo nhẹ (Warning Outline Button)
    WARNING_OUTLINE_BUTTON_STYLE = f"""
        QPushButton {{
            background-color: white;
            color: {WARNING_COLOR};
            border: 1px solid {WARNING_COLOR};
            border-radius: 4px;
            padding: {BUTTON_PADDING};
            height: {BUTTON_HEIGHT};
            {BOLD_FONT}
        }}
        
        QPushButton:hover {{
            background-color: #FFF8E1;
        }}
        
        QPushButton:pressed {{
            background-color: #FFECB3;
        }}
        
        QPushButton:disabled {{
            background-color: white;
            border-color: #BDBDBD;
            color: #BDBDBD;
        }}
    """
    
    # Nút màu phụ nhạt (Light Secondary Button)
    LIGHT_SECONDARY_BUTTON_STYLE = f"""
        QPushButton {{
            background-color: #E3F2FD;
            color: {SECONDARY_COLOR};
            border: none;
            border-radius: 4px;
            padding: {BUTTON_PADDING};
            height: {BUTTON_HEIGHT};
            {BOLD_FONT}
        }}
        
        QPushButton:hover {{
            background-color: #BBDEFB;
        }}
        
        QPushButton:pressed {{
            background-color: #90CAF9;
        }}
        
        QPushButton:disabled {{
            background-color: #F5F5F5;
            color: #BDBDBD;
        }}
    """
    
    # Nút tròn (Round Button)
    ROUND_BUTTON_STYLE = f"""
        QPushButton {{
            background-color: {PRIMARY_COLOR};
            color: white;
            border: none;
            border-radius: 20px;
            min-width: 40px;
            max-width: 40px;
            min-height: 40px;
            max-height: 40px;
            font-size: 18px;
            {BOLD_FONT}
        }}
        
        QPushButton:hover {{
            background-color: #303F9F;
        }}
        
        QPushButton:pressed {{
            background-color: #1A237E;
        }}
        
        QPushButton:disabled {{
            background-color: #BDBDBD;
            color: #757575;
        }}
    """
    
    PROCESS_BUTTON_STYLE = f"""
        QPushButton {{
            background-color: {SUCCESS_COLOR};
            color: white;
            border: none;
            border-radius: 4px;
            padding: 10px 15px;
            margin: 5px 0;
            {BOLD_FONT}
            font-size: 14px;
        }}
        
        QPushButton:hover {{
            background-color: #388E3C;
        }}
        
        QPushButton:pressed {{
            background-color: #2E7D32;
        }}
        
        QPushButton:disabled {{
            background-color: #BDBDBD;
            color: #757575;
        }}
    """
    
    # List widget style
    LIST_WIDGET_STYLE = f"""
        QListWidget {{
            border: 1px solid {BORDER_COLOR};
            border-radius: 4px;
            padding: 4px;
            background-color: white;
            outline: none;
        }}
        
        QListWidget::item {{
            border-radius: 2px;
            padding: 6px;
            margin: 2px 0;
        }}
        
        QListWidget::item:selected {{
            background-color: #E3F2FD;
            color: {PRIMARY_COLOR};
            {BOLD_FONT}
        }}
        
        QListWidget::item:hover:!selected {{
            background-color: #F5F5F5;
        }}
    """
    
    # Slider style
    SLIDER_STYLE = f"""
        QSlider::groove:horizontal {{
            border: 1px solid {BORDER_COLOR};
            height: 8px;
            background: white;
            border-radius: 4px;
        }}
        
        QSlider::handle:horizontal {{
            background: {SECONDARY_COLOR};
            border: 1px solid {SECONDARY_COLOR};
            width: 16px;
            height: 16px;
            margin: -5px 0;
            border-radius: 8px;
        }}
        
        QSlider::handle:horizontal:hover {{
            background: #1976D2;
            border: 1px solid #1976D2;
        }}
        
        QSlider::sub-page:horizontal {{
            background: #BBDEFB;
            border-radius: 4px;
        }}
    """
    
    # Text edit style
    TEXT_EDIT_STYLE = f"""
        QTextEdit {{
            border: 1px solid {BORDER_COLOR};
            border-radius: 4px;
            padding: 8px;
            background-color: white;
            selection-background-color: #BBDEFB;
        }}
        
        QTextEdit:focus {{
            border: 1px solid {SECONDARY_COLOR};
        }}
    """
    
    # Line edit style
    LINE_EDIT_STYLE = f"""
        QLineEdit {{
            border: 1px solid {BORDER_COLOR};
            border-radius: 4px;
            padding: 8px;
            background-color: white;
            selection-background-color: #BBDEFB;
        }}
        
        QLineEdit:focus {{
            border: 1px solid {SECONDARY_COLOR};
        }}
    """
    
    # Label styles
    HEADING_LABEL_STYLE = f"""
        QLabel {{
            {BOLD_FONT}
            color: {PRIMARY_COLOR};
            font-size: 14px;
            padding: 4px 0;
        }}
    """
    
    # Splitter style
    SPLITTER_STYLE = f"""
        QSplitter::handle {{
            background-color: {LIGHT_GRAY};
            border: 1px solid {BORDER_COLOR};
        }}
        
        QSplitter::handle:horizontal {{
            width: 4px;
        }}
        
        QSplitter::handle:vertical {{
            height: 4px;
        }}
        
        QSplitter::handle:hover {{
            background-color: {BORDER_COLOR};
        }}
    """
    
    # Scroll area style
    SCROLL_AREA_STYLE = f"""
        QScrollArea {{
            border: 1px solid {BORDER_COLOR};
            border-radius: 4px;
            background-color: white;
        }}
    """
    
    # Scrollbar style
    SCROLLBAR_STYLE = f"""
        QScrollBar:vertical {{
            border: none;
            background: {LIGHT_GRAY};
            width: 12px;
            margin: 0px;
            border-radius: 6px;
        }}
        
        QScrollBar::handle:vertical {{
            background: #BDBDBD;
            min-height: 20px;
            border-radius: 6px;
        }}
        
        QScrollBar::handle:vertical:hover {{
            background: #9E9E9E;
        }}
        
        QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{
            height: 0px;
        }}
        
        QScrollBar::up-arrow:vertical, QScrollBar::down-arrow:vertical {{
            background: none;
        }}
        
        QScrollBar::add-page:vertical, QScrollBar::sub-page:vertical {{
            background: none;
        }}
        
        QScrollBar:horizontal {{
            border: none;
            background: {LIGHT_GRAY};
            height: 12px;
            margin: 0px;
            border-radius: 6px;
        }}
        
        QScrollBar::handle:horizontal {{
            background: #BDBDBD;
            min-width: 20px;
            border-radius: 6px;
        }}
        
        QScrollBar::handle:horizontal:hover {{
            background: #9E9E9E;
        }}
        
        QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal {{
            width: 0px;
        }}
        
        QScrollBar::left-arrow:horizontal, QScrollBar::right-arrow:horizontal {{
            background: none;
        }}
        
        QScrollBar::add-page:horizontal, QScrollBar::sub-page:horizontal {{
            background: none;
        }}
    """
    
    # Group box style
    GROUP_BOX_STYLE = f"""
        QGroupBox {{
            {BOLD_FONT}
            border: 1px solid {BORDER_COLOR};
            border-radius: 4px;
            margin-top: 12px;
            padding-top: 16px;
        }}
        
        QGroupBox::title {{
            subcontrol-origin: margin;
            left: 10px;
            padding: 0 5px;
            color: {PRIMARY_COLOR};
        }}
    """
    
    # Message box style
    MESSAGE_BOX_STYLE = f"""
        QMessageBox {{
            background-color: white;
        }}
        
        QMessageBox QLabel {{
            {DEFAULT_FONT}
            color: {DARK_GRAY};
        }}
        
        QMessageBox QPushButton {{
            background-color: {SECONDARY_COLOR};
            color: white;
            border: none;
            border-radius: 4px;
            padding: {BUTTON_PADDING};
            min-width: 80px;
            {BOLD_FONT}
        }}
        
        QMessageBox QPushButton:hover {{
            background-color: #1976D2;
        }}
        
        QMessageBox QPushButton:pressed {{
            background-color: #0D47A1;
        }}
    """
    
    # Navigation button style (prev/next)
    NAV_BUTTON_STYLE = f"""
        QPushButton {{
            background-color: {PRIMARY_COLOR};
            color: white;
            border: none;
            border-radius: 4px;
            padding: {BUTTON_PADDING};
            height: {BUTTON_HEIGHT};
            min-width: 80px;
            {BOLD_FONT}
        }}
        
        QPushButton:hover {{
            background-color: #303F9F;
        }}
        
        QPushButton:pressed {{
            background-color: #1A237E;
        }}
        
        QPushButton:disabled {{
            background-color: #BDBDBD;
            color: #757575;
        }}
    """
    
    # Zoom control button style
    ZOOM_BUTTON_STYLE = f"""
        QPushButton {{
            background-color: {SECONDARY_COLOR};
            color: white;
            border: none;
            border-radius: 4px;
            font-size: 16px;
            font-weight: bold;
        }}
        
        QPushButton:hover {{
            background-color: #1976D2;
        }}
        
        QPushButton:pressed {{
            background-color: #0D47A1;
        }}
        
        QPushButton:disabled {{
            background-color: #BDBDBD;
            color: #757575;
        }}
    """
    
    # Tool button style (for ellipse operations)
    TOOL_BUTTON_STYLE = f"""
        QPushButton {{
            background-color: white;
            color: {DARK_GRAY};
            border: 1px solid {BORDER_COLOR};
            border-radius: 4px;
            padding: {BUTTON_PADDING};
            height: {BUTTON_HEIGHT};
        }}
        
        QPushButton:hover {{
            background-color: {LIGHT_GRAY};
            border-color: {SECONDARY_COLOR};
        }}
        
        QPushButton:pressed {{
            background-color: #E3F2FD;
            border-color: {SECONDARY_COLOR};
        }}
        
        QPushButton:disabled {{
            background-color: #F5F5F5;
            color: #BDBDBD;
            border-color: {BORDER_COLOR};
        }}
    """
    
    # Helper method to apply all styles to the application
    @staticmethod
    def apply_styles(app):
        """Apply all styles to the application"""
        style_sheet = (
            AnnotationStyles.APP_STYLE +
            AnnotationStyles.TAB_STYLE +
            AnnotationStyles.BUTTON_STYLE +
            AnnotationStyles.LIST_WIDGET_STYLE +
            AnnotationStyles.SLIDER_STYLE +
            AnnotationStyles.TEXT_EDIT_STYLE +
            AnnotationStyles.LINE_EDIT_STYLE +
            AnnotationStyles.SPLITTER_STYLE +
            AnnotationStyles.SCROLL_AREA_STYLE +
            AnnotationStyles.SCROLLBAR_STYLE +
            AnnotationStyles.GROUP_BOX_STYLE +
            AnnotationStyles.MESSAGE_BOX_STYLE
        )
        app.setStyleSheet(style_sheet)
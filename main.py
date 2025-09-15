import sys
import os
from PyQt5.QtWidgets import QApplication

from src.app import EllipseDetectorApp

if __name__ == "__main__":
    app = QApplication(sys.argv)
    
    window = EllipseDetectorApp()
    window.show()
    sys.exit(app.exec_())
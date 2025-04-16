#!/usr/bin/env python3
"""
Runner script for KC-Riff desktop application (PyQt6 version).
This script starts the PyQt6-based desktop interface that connects to our minimal server.
"""

import sys
import os

# Add the pyqt_interface directory to the Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
pyqt_interface_dir = os.path.join(current_dir, "pyqt_interface")
sys.path.append(pyqt_interface_dir)

# Import the launcher
from run_desktop_pyqt6 import main

if __name__ == "__main__":
    print("Starting KC-Riff Desktop Application (PyQt6 Version)")
    print("NOTE: Make sure the Minimal KC-Riff Server workflow is running first!")
    print("---------------------------------------------------------------")
    sys.exit(main())
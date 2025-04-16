#!/usr/bin/env python3
"""
Runner script for KC-Riff desktop application.
This script starts the PyQt5-based desktop interface that connects to our minimal server.
"""

import sys
import os
import subprocess
import signal
import time

# Add our pyqt_interface directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'pyqt_interface'))

# Import our desktop application
from pyqt_interface.kcriff_desktop import main as desktop_main

if __name__ == "__main__":
    print("Starting KC-Riff Desktop Application (PyQt5 Version)")
    print("NOTE: Make sure the Minimal KC-Riff Server workflow is running first!")
    print("NOTE: If you have PyQt6 installed, try using run_desktop_app_pyqt6.py instead!")
    print("---------------------------------------------------------------")
    
    # The Minimal KC-Riff Server should already be running as a workflow
    # Just start the desktop application directly
    try:
        desktop_main()
    except KeyboardInterrupt:
        print("\nShutting down...")
    except Exception as e:
        print(f"Error: {e}")
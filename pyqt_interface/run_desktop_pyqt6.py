#!/usr/bin/env python3
"""
Run the KC-Riff desktop application (PyQt6 version).
This script starts the PyQt6-based desktop interface for KC-Riff.
"""

import sys
import os
import subprocess
import time
import threading
import signal
import argparse
from kcriff_desktop_pyqt6 import main as desktop_main

def start_go_backend():
    """Start the Go backend server in a separate process"""
    print("Starting KC-Riff minimal backend...")
    # Start the server using the go executable - using minimal_server.go instead of kcriff.go
    try:
        process = subprocess.Popen(
            ["go", "run", "minimal_server.go"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        
        # Wait for the server to start
        stdout_ready = False
        for line in iter(process.stdout.readline, ''):
            print(f"[Backend] {line.strip()}")
            if "Starting minimal server on http://0.0.0.0:5000" in line:
                print("Backend server is running!")
                stdout_ready = True
                break
            
        # If we didn't see the startup message, check if the process is still running
        if not stdout_ready and process.poll() is not None:
            print(f"Backend process exited with code {process.returncode}")
            stderr_output = process.stderr.read()
            if stderr_output:
                print(f"stderr output: {stderr_output}")
            return None
        
        # Monitor stderr in a separate thread
        def monitor_stderr():
            for line in iter(process.stderr.readline, ''):
                print(f"[Backend Error] {line.strip()}")
        
        stderr_thread = threading.Thread(target=monitor_stderr)
        stderr_thread.daemon = True
        stderr_thread.start()
        
        return process
    except Exception as e:
        print(f"Error starting backend: {e}")
        return None

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='Run KC-Riff desktop application')
    parser.add_argument('--no-backend', action='store_true', help='Do not start the Go backend')
    args = parser.parse_args()
    
    backend_process = None
    
    try:
        # Start the backend if needed
        if not args.no_backend:
            backend_process = start_go_backend()
            if not backend_process:
                print("Failed to start backend. Exiting.")
                return 1
            
            # Give the server a moment to initialize
            time.sleep(1)
        
        # Start the desktop application
        print("Starting KC-Riff desktop interface with PyQt6...")
        desktop_main()
        
        return 0
    
    except KeyboardInterrupt:
        print("\nShutting down...")
    finally:
        # Clean up the backend process if it was started
        if backend_process:
            print("Stopping backend...")
            backend_process.send_signal(signal.SIGTERM)
            backend_process.wait()
            print("Backend stopped.")

if __name__ == "__main__":
    sys.exit(main())
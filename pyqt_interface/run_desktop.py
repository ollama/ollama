#!/usr/bin/env python3
"""
Run the KC-Riff desktop application.
This script starts the PyQt-based desktop interface for KC-Riff.
"""

import sys
import os
import subprocess
import time
import threading
import signal
import argparse
from kcriff_desktop import main as desktop_main

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
        for line in process.stdout:
            print(f"[Backend] {line.strip()}")
            if "Starting minimal server on http://0.0.0.0:5000" in line:
                print("Backend server is running!")
                break
        
        # Monitor stderr in a separate thread
        def monitor_stderr():
            for line in process.stderr:
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
        print("Starting KC-Riff desktop interface...")
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
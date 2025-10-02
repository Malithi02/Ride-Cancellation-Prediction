#!/usr/bin/env python3
"""
Ride Cancellation Prediction App Launcher with Ngrok
This script launches the Streamlit app and creates an ngrok tunnel
"""

import subprocess
import sys
import os
import time
import socket
from pathlib import Path

def is_port_open(port):
    """Check if a port is open"""
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.settimeout(2)
            return s.connect_ex(('127.0.0.1', port)) == 0
    except:
        return False

def wait_for_port(port, timeout=30):
    """Wait for a port to become available"""
    start_time = time.time()
    while time.time() - start_time < timeout:
        if is_port_open(port):
            return True
        time.sleep(1)
    return False

def main():
    """Launch the Streamlit app with ngrok"""
    # Get the directory where this script is located
    script_dir = Path(__file__).parent
    app_path = script_dir / "app" / "app.py"
    
    # Check if the app file exists
    if not app_path.exists():
        print(f"Error: App file not found at {app_path}")
        print("Please make sure you're running this from the project root directory.")
        sys.exit(1)
    
    # Check if models directory exists
    models_dir = script_dir / "models"
    if not models_dir.exists():
        print("Error: Models directory not found.")
        print("Please run the training notebook first to generate the model files.")
        sys.exit(1)
    
    print("Starting Ride Cancellation Prediction App with Ngrok...")
    print("=" * 60)
    
    # Set environment variable to skip Streamlit email prompt
    env = os.environ.copy()
    env['STREAMLIT_BROWSER_GATHER_USAGE_STATS'] = 'false'
    
    # Launch Streamlit in the background
    print("📱 Starting Streamlit app...")
    streamlit_process = subprocess.Popen([
        sys.executable, "-m", "streamlit", "run", 
        str(app_path),
        "--server.port", "8501",
        "--server.address", "localhost",
        "--server.headless", "true",
        "--browser.gatherUsageStats", "false"
    ], cwd=str(script_dir), env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    
    # Wait for Streamlit to start
    print("Waiting for Streamlit to start...")
    if wait_for_port(8501, 30):
        print("Streamlit app is running on http://localhost:8501")
    else:
        print("Streamlit failed to start within 30 seconds")
        streamlit_process.terminate()
        sys.exit(1)
    
    # Start ngrok
    print("Starting ngrok tunnel...")
    try:
        ngrok_process = subprocess.Popen([
            "ngrok", "http", "8501"
        ], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        
        # Give ngrok time to start
        time.sleep(3)
        
        print(" Ngrok tunnel created successfully!")
        print("Your app is now accessible via ngrok")
        print(" Local URL: http://localhost:8501")
        print("Public URL: Check ngrok dashboard at http://localhost:4040")
        print("=" * 60)
        print("Press Ctrl+C to stop both services")
        
        # Keep the script running
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            print("\nShutting down services...")
            streamlit_process.terminate()
            ngrok_process.terminate()
            print("👋 Services stopped")
            
    except FileNotFoundError:
        print(" Ngrok not found. Please install ngrok first:")
        print("   1. Download from https://ngrok.com/download")
        print("   2. Add to your PATH")
        print("   3. Run: ngrok authtoken YOUR_TOKEN")
        print("\n📱 App is still running locally at: http://localhost:8501")
        
        # Keep Streamlit running
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            print("\n Shutting down Streamlit...")
            streamlit_process.terminate()
            print("👋 Streamlit stopped")

if __name__ == "__main__":
    main()

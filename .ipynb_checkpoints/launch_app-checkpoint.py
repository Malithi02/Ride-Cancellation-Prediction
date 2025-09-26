#!/usr/bin/env python3
"""
Ride Cancellation Prediction App Launcher
This script launches the Streamlit app with proper configuration
"""

import subprocess
import sys
import os
from pathlib import Path

def main():
    """Launch the Streamlit app"""
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
    
    # Check if required model files exist
    required_files = ['best_model.pkl', 'scaler.pkl', 'ohe.pkl', 'le_dict.pkl']
    missing_files = []
    for file in required_files:
        if not (models_dir / file).exists():
            missing_files.append(file)
    
    if missing_files:
        print(f"Error: Missing model files: {', '.join(missing_files)}")
        print("Please run the training notebook first to generate the model files.")
        sys.exit(1)
    
    print("🚗 Starting Ride Cancellation Prediction App...")
    print("=" * 50)
    print("App will be available at: http://localhost:8501")
    print("Press Ctrl+C to stop the app")
    print("=" * 50)
    
    try:
        # Set environment variable to skip Streamlit email prompt
        env = os.environ.copy()
        env['STREAMLIT_BROWSER_GATHER_USAGE_STATS'] = 'false'
        
        # Launch Streamlit with non-interactive mode
        subprocess.run([
            sys.executable, "-m", "streamlit", "run", 
            str(app_path),
            "--server.port", "8501",
            "--server.address", "localhost",
            "--server.headless", "true",
            "--browser.gatherUsageStats", "false"
        ], cwd=str(script_dir), env=env)
    except KeyboardInterrupt:
        print("\n👋 App stopped by user")
    except Exception as e:
        print(f"Error launching app: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()

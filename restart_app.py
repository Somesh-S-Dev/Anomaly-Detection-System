#!/usr/bin/env python3
"""
Script to restart the Flask application with cache clearing.
"""

import os
import sys
import subprocess
import time
import tempfile

def clear_cache():
    """Clear all cached files."""
    print("🧹 Clearing cache files...")
    
    # Clear Flask cache
    temp_file = os.path.join(tempfile.gettempdir(), 'anomaly_detection_results.pkl')
    models_file = os.path.join(tempfile.gettempdir(), 'anomaly_detection_models.pkl')
    
    for file_path in [temp_file, models_file]:
        if os.path.exists(file_path):
            try:
                os.remove(file_path)
                print(f"✅ Removed: {file_path}")
            except Exception as e:
                print(f"❌ Error removing {file_path}: {e}")
    
    # Clear Python cache
    for root, dirs, files in os.walk('.'):
        for dir_name in dirs:
            if dir_name == '__pycache__':
                cache_dir = os.path.join(root, dir_name)
                try:
                    import shutil
                    shutil.rmtree(cache_dir)
                    print(f"✅ Removed cache directory: {cache_dir}")
                except Exception as e:
                    print(f"❌ Error removing cache directory {cache_dir}: {e}")

def restart_app():
    """Restart the Flask application."""
    print("🚀 Restarting Flask application...")
    
    # Kill existing Flask process if running
    try:
        subprocess.run(['taskkill', '/f', '/im', 'python.exe'], 
                      capture_output=True, check=False)
        print("✅ Killed existing Python processes")
    except Exception as e:
        print(f"⚠️ Could not kill existing processes: {e}")
    
    # Wait a moment
    time.sleep(2)
    
    # Start the application
    try:
        print("🔄 Starting Flask application...")
        subprocess.Popen([sys.executable, 'app.py'], 
                        stdout=subprocess.PIPE, 
                        stderr=subprocess.PIPE)
        print("✅ Flask application started!")
        print("🌐 Open your browser and go to: http://127.0.0.1:5000")
        print("📝 Remember to:")
        print("   1. Clear browser cache (Ctrl+F5)")
        print("   2. Run models again")
        print("   3. Check Model Comparison page for accuracy column")
    except Exception as e:
        print(f"❌ Error starting Flask application: {e}")

if __name__ == "__main__":
    print("🔄 Flask Application Restart Tool")
    print("=" * 40)
    
    clear_cache()
    print()
    restart_app() 
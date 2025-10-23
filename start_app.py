#!/usr/bin/env python3
"""
Simple startup script for the anomaly detection application
"""

import os
import sys
import subprocess

def check_dependencies():
    """Check if required dependencies are available."""
    required_packages = [
        'flask', 'pandas', 'numpy', 'scikit-learn', 
        'torch', 'plotly', 'imbalanced-learn'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
            print(f"✅ {package}")
        except ImportError:
            missing_packages.append(package)
            print(f"❌ {package} - MISSING")
    
    if missing_packages:
        print(f"\n⚠️  Missing packages: {', '.join(missing_packages)}")
        print("Please install missing packages using: pip install -r requirements.txt")
        return False
    
    return True

def main():
    """Main startup function."""
    print("🚀 Starting Anomaly Detection Application...")
    print("="*50)
    
    # Check dependencies
    print("📦 Checking dependencies...")
    if not check_dependencies():
        print("\n❌ Dependencies check failed. Please install missing packages.")
        return False
    
    print("\n✅ All dependencies available!")
    
    # Check if data file exists
    data_file = 'SSBCI-Transactions-Dataset.csv'
    if not os.path.exists(data_file):
        print(f"\n❌ Data file '{data_file}' not found!")
        print("Please ensure the dataset file is in the project directory.")
        return False
    
    print(f"✅ Data file '{data_file}' found!")
    
    # Start the application
    print("\n🌐 Starting Flask application...")
    print("📍 Server will be available at: http://127.0.0.1:5000")
    print("🛑 Press Ctrl+C to stop the server")
    print("="*50)
    
    try:
        # Import and run the app
        from app import app
        app.run(
            host='127.0.0.1',
            port=5000,
            debug=False,
            use_reloader=False,
            threaded=True
        )
        return True
        
    except KeyboardInterrupt:
        print("\n🛑 Server stopped by user")
        return True
    except Exception as e:
        print(f"\n❌ Error starting application: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    if not success:
        sys.exit(1) 
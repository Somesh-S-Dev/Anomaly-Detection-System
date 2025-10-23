@echo off
echo ========================================
echo    Anomaly Detection Application
echo ========================================
echo.

REM Check if virtual environment exists
if not exist "ADP\Scripts\activate.bat" (
    echo Error: Virtual environment not found!
    echo Please run: python -m venv ADP
    echo Then run: ADP\Scripts\activate
    pause
    exit /b 1
)

REM Activate virtual environment
echo Activating virtual environment...
call ADP\Scripts\activate.bat

REM Check if requirements are installed
echo Checking dependencies...
python -c "import flask, pandas, numpy, sklearn, torch" 2>nul
if errorlevel 1 (
    echo Installing requirements...
    pip install -r requirements.txt
)

REM Start the application
echo.
echo Starting Flask application...
echo Server will be available at: http://127.0.0.1:5000
echo Press Ctrl+C to stop the server
echo.
python start_app.py

pause 
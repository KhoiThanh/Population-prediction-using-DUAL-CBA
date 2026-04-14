@echo off
REM Script to run Vietnam Population Forecast Demo App
REM Windows Batch File

echo.
echo =========================================
echo Vietnam Population Forecast Demo App
echo =========================================
echo.

REM Check if streamlit is installed
python -m pip show streamlit > nul 2>&1
if %errorlevel% neq 0 (
    echo Installing required packages...
    python -m pip install -r requirements_demo.txt
)

echo.
echo Starting Streamlit app...
echo Opening http://localhost:8501 in your browser
echo.
python -m streamlit run demo_vietnam_population.py

pause

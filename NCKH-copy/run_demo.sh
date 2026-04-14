#!/bin/bash
# Script to run Vietnam Population Forecast Demo App
# Unix/Linux/Mac Shell Script

echo ""
echo "========================================="
echo "Vietnam Population Forecast Demo App"
echo "========================================="
echo ""

# Check if streamlit is installed
python3 -m pip show streamlit > /dev/null 2>&1
if [ $? -ne 0 ]; then
    echo "Installing required packages..."
    python3 -m pip install -r requirements_demo.txt
fi

echo ""
echo "Starting Streamlit app..."
echo "Opening http://localhost:8501 in your browser"
echo ""

python3 -m streamlit run demo_vietnam_population.py

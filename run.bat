@echo off
echo Installing dependencies...
pip install flask numpy scipy matplotlib pillow
echo.
echo Starting Diffraction Simulator...
echo Open browser at http://localhost:5000
echo.
python app.py
pause
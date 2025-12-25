@echo off
cd /d "%~dp0"
if not exist "templates" mkdir templates
if not exist "static" mkdir static
pip install flask numpy pillow
start http://127.0.0.1:5000
python app.py
pause
@echo off
chcp 65001 >nul

pip install flask numpy scipy matplotlib opencv-python pillow imageio[ffmpeg] >nul 2>&1

start http://localhost:5000
python app.py
pause
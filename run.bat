@echo off
chcp 65001 >nul
cd /d "%~dp0"

echo ==========================================
echo   Симулятор дифракции электронов
echo ==========================================
echo.

python --version >nul 2>&1
if errorlevel 1 (
    echo ОШИБКА: Python не найден!
    echo Установите Python 3.8+ с python.org
    pause
    exit /b 1
)

if not exist "templates" mkdir templates
if not exist "static" mkdir static

echo Установка зависимостей...
pip install flask numpy scipy pillow matplotlib --quiet

echo.
echo Запуск сервера на http://127.0.0.1:5000
echo Нажмите Ctrl+C для остановки
echo.
start http://127.0.0.1:5000
python app.py

pause
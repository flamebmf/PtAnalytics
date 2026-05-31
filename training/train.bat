@echo off
setlocal
cd /d "%~dp0"

set PY="%LOCALAPPDATA%\Programs\Python\Python312\python.exe"

if not exist %PY% (
    echo Python 3.12 not found at %PY%
    echo Install Python 3.12 first
    pause
    exit /b 1
)

echo ========================================
echo  CAM Training Tool
echo ========================================
echo.
echo  1  Train (640px, skip already trained)
echo  2  Train (1280px, skip already trained)
echo  3  Retrain all (640px)
echo  4  Retrain all (1280px)
echo  5  Status (show trained datasets)
echo  0  Exit
echo.
set /p choice="Choose [0-5]: "

if "%choice%"=="1" %PY% train.py 640
if "%choice%"=="2" %PY% train.py 1280
if "%choice%"=="3" %PY% train.py 640 --force
if "%choice%"=="4" %PY% train.py 1280 --force
if "%choice%"=="5" %PY% -c "import json; s=json.load(open('.train-state.json')); print('\nTrained:'); [print(f'  {k}: {v[\"date\"]} @ {v.get(\"imgsz\",\"?\")}px') for k,v in s.items()] if s else print('  nothing yet')"
if "%choice%"=="0" exit /b 0

echo.
echo Done.
pause

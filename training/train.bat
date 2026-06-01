# Copyright (c) 2026 PlurumTech.com
# SPDX-License-Identifier: GPL-3.0-only
@echo off
setlocal enabledelayedexpansion
cd /d "%~dp0"

set PY="%LOCALAPPDATA%\Programs\Python\Python312\python.exe"

if not exist %PY% (
    echo Python 3.12 not found at %PY%
    echo Install Python 3.12 first
    pause
    exit /b 1
)

:menu
echo ========================================
echo  CAM Training Tool
echo ========================================
echo.
echo  1  Train (640px, skip already trained)
echo  2  Train (1280px, skip already trained)
echo  3  Retrain all (640px)
echo  4  Retrain all (1280px)
echo  5  Status (show trained datasets)
echo  6  Download ZIP from server ^& train (640px)
echo  7  Cleanup old artifacts (per-name models, extracted dataset)
echo  0  Exit
echo.
set /p choice="Choose [0-7]: "

if "%choice%"=="1" %PY% train.py 640
if "%choice%"=="2" %PY% train.py 1280
if "%choice%"=="3" %PY% train.py 640 --force
if "%choice%"=="4" %PY% train.py 1280 --force
if "%choice%"=="5" %PY% -c "import json; s=json.load(open('.train-state.json')); print('\nTrained:'); [print(f'  {k}: {v[\"date\"]} @ {v.get(\"imgsz\",\"?\")}px') for k,v in s.items()] if s else print('  nothing yet')"
if "%choice%"=="6" goto download_and_train
if "%choice%"=="7" goto cleanup
if "%choice%"=="0" exit /b 0

:download_and_train
echo.
echo Enter server URL (without trailing slash):
set /p server_url="URL (e.g. http://192.168.1.100:8090): "
echo Downloading combined dataset from %server_url%/training/export ...
%PY% -c "
import urllib.request, sys, os
url = sys.argv[1] + '/training/export'
zip_path = os.path.join('extracted', 'combined.zip')
os.makedirs('extracted', exist_ok=True)
urllib.request.urlretrieve(url, zip_path)
print(f'Downloaded to {zip_path}')
" "!server_url!"
%PY% train.py 640 --force
pause
goto menu

:cleanup
echo.
echo Cleaning up old artifacts...
for %%d in (im lena mazda) do (
    if exist "models\%%d" (
        rmdir /s /q "models\%%d"
        echo  - models\%%d
    )
    if exist "fine-tuned-%%d.pt" (
        del "fine-tuned-%%d.pt"
        echo  - fine-tuned-%%d.pt
    )
)
if exist "extracted" (
    rmdir /s /q "extracted"
    echo  - extracted\
)
echo Done.
pause
goto menu

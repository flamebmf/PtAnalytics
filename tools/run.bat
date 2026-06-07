@echo off
chcp 65001 >nul
cd /d "%~dp0.."
"C:\Users\dakulich\AppData\Local\Programs\Python\Python312\python.exe" tools\run.py %*
pause

@echo off
chcp 65001 >nul
title UtaiSynthesizer 开发模式

cd /d "%~dp0"

echo ======================================
echo   UtaiSynthesizer 开发模式启动中...
echo ======================================
echo.

npm run tauri dev

pause

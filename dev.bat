@echo off
chcp 65001 >nul 2>&1
title Muno Dev — 开发模式 (Vite HMR)
cd /d "%~dp0"
echo.
echo ╔══════════════════════════════════════╗
echo ║   Muno 开发者模式 - 热更新已启用      ║
echo ║   改代码 → Ctrl+S → 自动刷新 UI       ║
echo ╚══════════════════════════════════════╝
echo.
echo 前端 dev server: http://localhost:1420
echo HMR WebSocket  : ws://localhost:1421
echo.
echo 关闭此窗口 = 停止开发模式
echo.
npm run tauri dev
pause

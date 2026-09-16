@echo off
rem ===========================================================================
rem  面板：双击运行，面板在这个窗口里前台跑，关掉窗口就是停止面板。
rem
rem  要改端口 / 设备 / 解释器，改下面「配置」那三行。
rem ===========================================================================
title Graph-Junction-Diffusion 面板
cd /d "%~dp0.."

rem ---- 配置 ------------------------------------------------------------------
set "PYTHON=E:\CondaEnvData\envs\GGMPC\python.exe"
set "PORT=8765"
set "DEVICE=cpu"

echo.
echo   正在启动面板...
echo   地址        http://127.0.0.1:%PORT%/
echo   停止方式    关掉这个窗口
echo.

"%PYTHON%" dashboard\server.py --port %PORT% --device %DEVICE%

echo.
echo   面板已退出，上面若有报错就是原因。
pause

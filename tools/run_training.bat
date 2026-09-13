@echo off
REM 100 epoch 正式训练（Controlled Junction 数据集）
REM 用法：cmd /c tools\run_training.bat [--resume <ckpt> --extra-epochs N]
set PYTHONIOENCODING=utf-8
set PYTHONUNBUFFERED=1
cd /d "%~dp0.."
echo [run] start %DATE% %TIME%  args=%*
"E:\CondaEnvData\envs\GGMPC\python.exe" scripts/train.py ^
  --config configs/graph_flow.yaml ^
  --name v2_controlled_100ep ^
  --data data/controlled_train.pkl ^
  --val-data data/controlled_val.pkl ^
  %*
echo [run] exit code %ERRORLEVEL% at %DATE% %TIME%

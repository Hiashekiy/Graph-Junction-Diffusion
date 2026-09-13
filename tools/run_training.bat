@echo off
REM 100 epoch 正式训练（Controlled Junction 数据集）
REM 用法：cmd /c tools\run_training.bat [--resume <ckpt> --extra-epochs N]
REM
REM 注意：这里**不**传 --name，run 名由 configs/graph_flow.yaml 的 paths.run_name 决定。
REM 硬编码名字很危险：改了 flow_steps 之后新 run 会直接覆盖旧 run 的 best.pt/history.json。
set PYTHONIOENCODING=utf-8
set PYTHONUNBUFFERED=1
cd /d "%~dp0.."
echo [run] start %DATE% %TIME%  args=%*
"E:\CondaEnvData\envs\GGMPC\python.exe" scripts/train.py ^
  --config configs/graph_flow.yaml ^
  --data data/controlled_train.pkl ^
  --val-data data/controlled_val.pkl ^
  %*
echo [run] exit code %ERRORLEVEL% at %DATE% %TIME%

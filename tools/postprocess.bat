@echo off
REM Post-training evaluation pipeline:
REM   test eval -> curve summary -> bucket breakdown -> paired tests -> validation paired -> ablation
REM
REM Usage: cmd /c tools\postprocess.bat [run_name] [baseline_run_name]
REM   defaults: run_name=v2_controlled_100ep_flow3, baseline_run_name=v2_controlled_100ep
REM
REM NOTE (why comments here are ASCII only): cmd.exe parses .bat files line by line and
REM mishandles non-ASCII text in REM lines when the file has LF line endings -- the lines
REM get re-split and cmd tries to execute fragments of them. Keep this file ASCII and keep
REM it checked out with CRLF (see .gitattributes).
REM
REM NOTE: the ablation and the timing numbers must be measured with nothing else using the
REM GPU, otherwise sec/query is inflated (measured 0.024 -> 0.042 s/query). Run this script
REM after the training process has exited.
setlocal
set PYTHONIOENCODING=utf-8
set PYTHONUNBUFFERED=1
cd /d "%~dp0.."
set "PY=E:\CondaEnvData\envs\GGMPC\python.exe"
set "RUN=%~1"
if "%RUN%"=="" set "RUN=v2_controlled_100ep_flow3"
set "BASE=%~2"
if "%BASE%"=="" set "BASE=v2_controlled_100ep"

echo [postprocess] run=%RUN% baseline=%BASE% start %DATE% %TIME%

echo.
echo [1/6] summarize %RUN% on the test split (writes eval_test.json + summary.json/txt)
"%PY%" tools/summarize_run.py outputs/runs/%RUN% ^
  --test-data data/controlled_test.pkl --baselines
if errorlevel 1 goto :failed

echo.
echo [2/6] difficulty / mode breakdown
"%PY%" tools/breakdown_eval.py outputs/runs/%RUN%/eval_test.json ^
  --data data/controlled_test.pkl --out outputs/runs/%RUN%/breakdown_test.json
if errorlevel 1 goto :failed

echo.
echo [3/6] paired comparison vs %BASE% (goal_hit / optimal / broken)
"%PY%" tools/paired_compare.py ^
  --a outputs/runs/%BASE%/eval_test.json --b outputs/runs/%RUN%/eval_test.json ^
  --data data/controlled_test.pkl --metric goal_hit ^
  --label-a %BASE% --label-b %RUN% --out outputs/runs/%RUN%/paired_goal_hit.json
"%PY%" tools/paired_compare.py ^
  --a outputs/runs/%BASE%/eval_test.json --b outputs/runs/%RUN%/eval_test.json ^
  --data data/controlled_test.pkl --metric optimal ^
  --label-a %BASE% --label-b %RUN% --out outputs/runs/%RUN%/paired_optimal.json
"%PY%" tools/paired_compare.py ^
  --a outputs/runs/%BASE%/eval_test.json --b outputs/runs/%RUN%/eval_test.json ^
  --data data/controlled_test.pkl --metric broken ^
  --label-a %BASE% --label-b %RUN% --out outputs/runs/%RUN%/paired_broken.json
if errorlevel 1 goto :failed

echo.
echo [4/6] paired comparison on the validation split at matched epochs
for %%E in (25 30 35 40 45 50 55 60 65 70 75 80 85 90 95 100) do (
  if exist outputs/runs/%RUN%/val_records_epoch%%E.json (
    echo   -- epoch %%E
    "%PY%" tools/paired_compare.py ^
      --a outputs/runs/%BASE%/val_records_epoch%%E.json ^
      --b outputs/runs/%RUN%/val_records_epoch%%E.json ^
      --data data/controlled_val.pkl --metric goal_hit ^
      --label-a %BASE% --label-b %RUN%
  )
)

echo.
echo [5/6] training-curve comparison (aligned by true epoch)
"%PY%" tools/compare_curves.py ^
  --a outputs/runs/%BASE% --b outputs/runs/%RUN% ^
  --label-a %BASE% --label-b %RUN% --every 5 ^
  --out outputs/runs/%RUN%/curves_vs_baseline.json

echo.
echo [6/6] inference flow_steps ablation (1..trained rounds)
"%PY%" tools/ablation_eval.py outputs/runs/%RUN% ^
  --data data/controlled_test.pkl --no-progress
if errorlevel 1 goto :failed

echo.
echo [postprocess] done %DATE% %TIME%
exit /b 0

:failed
echo.
echo [postprocess] FAILED (exit code %ERRORLEVEL%) at %DATE% %TIME%
exit /b 1

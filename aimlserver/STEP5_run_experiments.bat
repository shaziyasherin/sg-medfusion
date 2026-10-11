@echo off
rem THE GPU STEP. 12 experiments x 3 repetitions (only 8 use the GPU).
rem You can STOP at any time: just close this window. Run this file again later
rem and it continues (finished runs are kept, an unfinished run loses at most 5 rounds).
call "%~dp0_settings.bat"
powershell -NoProfile -Command "python -X utf8 scripts\run_all.py --config %CONFIG% --stage run --plan minimal --seeds 0 1 2 2>&1 | Tee-Object -FilePath logs\step5_run.log -Append"
pause

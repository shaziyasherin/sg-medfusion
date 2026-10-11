@echo off
rem Shows how many runs are finished and the time left. Safe to use any time.
call "%~dp0_settings.bat"
python scripts\progress.py --config %CONFIG% --plan minimal --seeds 0 1 2
pause

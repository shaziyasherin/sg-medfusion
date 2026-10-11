@echo off
rem CPU only, 2 minutes. Makes tables.md and the figures from the finished runs.
call "%~dp0_settings.bat"
python -X utf8 scripts\make_report.py %CONFIG%
echo.
echo Copy D:\sgmedfusion\results to the laptop, but WITHOUT the .npz and .ckpt files.
pause

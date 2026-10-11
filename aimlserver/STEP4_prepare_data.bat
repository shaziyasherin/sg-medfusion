@echo off
rem CPU only, about 1 minute. Reads the eICU tables once.
call "%~dp0_settings.bat"
powershell -NoProfile -Command "python -X utf8 scripts\run_all.py --config %CONFIG% --stage extract 2>&1 | Tee-Object -FilePath logs\step4_prepare.log"
echo.
echo The last line must say: wrote D:\sgmedfusion\work\timelines.pkl
pause

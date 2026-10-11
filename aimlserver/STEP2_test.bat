@echo off
rem 10 minutes, CPU only. Checks the code works.
call "%~dp0_settings.bat"
python -X utf8 tests\test_pipeline.py > logs\step2_test.log 2>&1
findstr /B "ok" logs\step2_test.log
echo.
echo You must see 6 lines starting with "ok". If not, send logs\step2_test.log to Claude.
pause

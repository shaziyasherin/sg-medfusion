@echo off
rem Downloads the open eICU demo (about 21 MB, no login). UK-DALE is already downloaded.
rem If it stops, run it again - finished files are skipped.
call "%~dp0_settings.bat"
python scripts\download_eicu.py --out D:\sgmedfusion\data\eicu_demo
python scripts\download_ukdale.py --out D:\sgmedfusion\data\ukdale
pause

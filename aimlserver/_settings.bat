@echo off
set CODE=D:\sgmedfusion\code
set CONFIG=configs\paths_aimlserver.json
call conda activate sgmf
cd /d %CODE%
if not exist logs mkdir logs

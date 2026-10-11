@echo off
rem ONE TIME ONLY (15 min). Creates a separate Python environment called sgmf.
call conda create -y -n sgmf python=3.10
call conda activate sgmf
pip install torch --index-url https://download.pytorch.org/whl/cu121
cd /d D:\sgmedfusion\code
pip install -r requirements.txt
python -c "import torch; print('RESULT:', 'GPU OK' if torch.cuda.is_available() else 'NO GPU - tell Claude', torch.cuda.get_device_name(0) if torch.cuda.is_available() else '')"
pause

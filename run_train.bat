@echo off
python .\script\train.py --batch-size 2 --epochs 1 --num-workers 0 --finetune
pause 
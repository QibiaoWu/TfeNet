# !/bin/bash -e

# python main.py --model TfeNet -b 1 --save-dir TfeNet \
# --cubesize 128 128 128 --stridet 64 64 64 --stridev 64 64 64 --worker 4 \
# --start-epoch 1 --epoch 60 --sgd 1  --resumepart 0 --device 1 \
# --test 1 --resume ./checkpoint/TfeNet_checkpoint.ckpt

python evaluation.py

python concat.py

python postprocessing.py



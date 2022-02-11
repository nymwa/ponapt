#!/bin/bash

time python infer.py \
        --checkpoint checkpoints/lm.pt \
        --hidden-dim 1024 \
        --nhead 16 \
        --feedforward-dim 4096 \
        --num-layers 6 \
        --iters 1000 \
        | tee aug.txt

#!/bin/bash

python train.py \
        --share-embedding \
        | tee train.log


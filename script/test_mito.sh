#!/usr/bin/env bash

python test/eval_mito.py \
--gpu 3 \
--exp FS_003_bs16 \
--num-workers 8 \
--batch-size 1 \
--num-classes 4 \
--test_idx 'iso'
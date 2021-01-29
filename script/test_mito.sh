#!/usr/bin/env bash

python test/eval_mito.py \
--gpu 0 \
--exp FS_mito \
--num-workers 8 \
--batch-size 1 \
--num-classes 4 \
--test_idx 'iso'
#!/usr/bin/env bash


python experiments/test_mem_nu.py \
--gpu 0 \
--exp FS_mem_nu \
--num-workers 8 \
--batch-size 1 \
--num-classes 2 \
--test_idx 'iso' \
--mode 'mem_nu' \


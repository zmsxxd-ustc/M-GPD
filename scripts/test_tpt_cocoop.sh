#!/bin/bash

#data_root='/data/linshiqi047'
data_root='/data/zmsxxd'
coop_weight='/path/to/pretrained/coop/weight.pth'
testsets=$1
arch=/model/zmsxxd/RN50/ViT-B-16.pt
arch_t=/model/zmsxxd/RN50/ViT-L-14.pt #ViT-L-14.pt
# arch=ViT-B/16
bs=64

python ./tpt_classification.py ${data_root} --test_sets ${testsets} \
--arch ${arch} --arch_t ${arch_t} -b ${bs} --gpu 0 \
--tpt --cocoop --lr 0.01 --BETA 0.9 --n_ctx 4
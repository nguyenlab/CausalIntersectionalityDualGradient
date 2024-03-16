#!/bin/bash

# OV50
## Loading finetuned without having to move it to ./data
loadfin=${1:-./data/LASTtrain.pth}
loadfin2=${2:-./data/LASTtraindev.pth}
loadfin3=${3:-./data/LASTtraindev.pth}

## 50 VG feats, Seed 84
cp ./data/hm_vg5050.tsv ./data/HM_img.tsv

### attattr
python hm.py --seed 84 --model O \
    --test dev_seen --lr 1e-5 --batchSize 8 --tr bert-large-uncased --epochs 5 --tsv \
    --num_features 50 --loadfin $loadfin --exp OV50 \
    --num_steps 10 --output /content/drive/MyDrive/vilio/export/O/attattr --extract_type attattr \
    --conf_modality txt --conf_unit_size 2 --preds_threshold .0 --highlight_sample

python hm.py --seed 84 --model O \
    --test dev_seen --lr 1e-5 --batchSize 8 --tr bert-large-uncased --epochs 5 --tsv \
    --num_features 50 --loadfin $loadfin --exp OV50 \
    --num_steps 10 --output /content/drive/MyDrive/vilio/export/O/attattr --extract_type attattr \
    --conf_modality img --conf_unit_size 2 --preds_threshold .0 --highlight_sample

### attention
python hm.py --seed 84 --model O \
    --test dev_seen --lr 1e-5 --batchSize 8 --tr bert-large-uncased --epochs 5 --tsv \
    --num_features 50 --loadfin $loadfin --exp OV50 \
    --num_steps 10 --output /content/drive/MyDrive/vilio/export/O/attention --extract_type attention \
    --conf_modality txt --conf_unit_size 2 --preds_threshold .0

python hm.py --seed 84 --model O \
    --test dev_seen --lr 1e-5 --batchSize 8 --tr bert-large-uncased --epochs 5 --tsv \
    --num_features 50 --loadfin $loadfin --exp OV50 \
    --num_steps 10 --output /content/drive/MyDrive/vilio/export/O/attention --extract_type attention \
    --conf_modality img --conf_unit_size 2 --preds_threshold .0

### gradient
python hm.py --seed 84 --model O \
    --test dev_seen --lr 1e-5 --batchSize 8 --tr bert-large-uncased --epochs 5 --tsv \
    --num_features 50 --loadfin $loadfin --exp OV50 \
    --num_steps 10 --output /content/drive/MyDrive/vilio/export/O/gradient --extract_type gradient \
    --conf_modality txt --conf_unit_size 2 --preds_threshold .0

python hm.py --seed 84 --model O \
    --test dev_seen --lr 1e-5 --batchSize 8 --tr bert-large-uncased --epochs 5 --tsv \
    --num_features 50 --loadfin $loadfin --exp OV50 \
    --num_steps 10 --output /content/drive/MyDrive/vilio/export/O/gradient --extract_type gradient \
    --conf_modality img --conf_unit_size 2 --preds_threshold .0

## 50 feats, Seed 126
cp ./data/hm_vgattr5050.tsv ./data/HM_img.tsv

### attattr
python hm.py --seed 126 --model O \
    --test dev_seen --lr 1e-5 --batchSize 8 --tr bert-large-uncased --epochs 5 --tsv \
    --num_features 50 --loadfin $loadfin2 --exp O50 \
    --num_steps 10 --output /content/drive/MyDrive/vilio/export/O/attattr --extract_type attattr \
    --conf_modality txt --conf_unit_size 2 --preds_threshold .0 --highlight_sample

python hm.py --seed 126 --model O \
    --test dev_seen --lr 1e-5 --batchSize 8 --tr bert-large-uncased --epochs 5 --tsv \
    --num_features 50 --loadfin $loadfin2 --exp O50 \
    --num_steps 10 --output /content/drive/MyDrive/vilio/export/O/attattr --extract_type attattr \
    --conf_modality img --conf_unit_size 2 --preds_threshold .0 --highlight_sample

### attention
python hm.py --seed 126 --model O \
    --test dev_seen --lr 1e-5 --batchSize 8 --tr bert-large-uncased --epochs 5 --tsv \
    --num_features 50 --loadfin $loadfin2 --exp O50 \
    --num_steps 10 --output /content/drive/MyDrive/vilio/export/O/attention --extract_type attention \
    --conf_modality txt --conf_unit_size 2 --preds_threshold .0

python hm.py --seed 126 --model O \
    --test dev_seen --lr 1e-5 --batchSize 8 --tr bert-large-uncased --epochs 5 --tsv \
    --num_features 50 --loadfin $loadfin2 --exp O50 \
    --num_steps 10 --output /content/drive/MyDrive/vilio/export/O/attention --extract_type attention \
    --conf_modality img --conf_unit_size 2 --preds_threshold .0

### gradient
python hm.py --seed 126 --model O \
    --test dev_seen --lr 1e-5 --batchSize 8 --tr bert-large-uncased --epochs 5 --tsv \
    --num_features 50 --loadfin $loadfin2 --exp O50 \
    --num_steps 10 --output /content/drive/MyDrive/vilio/export/O/gradient --extract_type gradient \
    --conf_modality txt --conf_unit_size 2 --preds_threshold .0

python hm.py --seed 126 --model O \
    --test dev_seen --lr 1e-5 --batchSize 8 --tr bert-large-uncased --epochs 5 --tsv \
    --num_features 50 --loadfin $loadfin2 --exp O50 \
    --num_steps 10 --output /content/drive/MyDrive/vilio/export/O/gradient --extract_type gradient \
    --conf_modality img --conf_unit_size 2 --preds_threshold .0

## 36 Feats, Seed 42
cp ./data/hm_vgattr3636.tsv ./data/HM_img.tsv

### attattr
python hm.py --seed 42 --model O \
    --test dev_seen --lr 1e-5 --batchSize 8 --tr bert-large-uncased --epochs 5 --tsv \
    --num_features 36 --loadfin $loadfin3 --exp O36 \
    --num_steps 10 --output /content/drive/MyDrive/vilio/export/O/attattr --extract_type attattr \
    --conf_modality txt --conf_unit_size 2 --preds_threshold .0 --highlight_sample

python hm.py --seed 42 --model O \
    --test dev_seen --lr 1e-5 --batchSize 8 --tr bert-large-uncased --epochs 5 --tsv \
    --num_features 36 --loadfin $loadfin3 --exp O36 \
    --num_steps 10 --output /content/drive/MyDrive/vilio/export/O/attattr --extract_type attattr \
    --conf_modality img --conf_unit_size 2 --preds_threshold .0 --highlight_sample

### attention
python hm.py --seed 42 --model O \
    --test dev_seen --lr 1e-5 --batchSize 8 --tr bert-large-uncased --epochs 5 --tsv \
    --num_features 36 --loadfin $loadfin3 --exp O36 \
    --num_steps 10 --output /content/drive/MyDrive/vilio/export/O/attention --extract_type attention \
    --conf_modality txt --conf_unit_size 2 --preds_threshold .0

python hm.py --seed 42 --model O \
    --test dev_seen --lr 1e-5 --batchSize 8 --tr bert-large-uncased --epochs 5 --tsv \
    --num_features 36 --loadfin $loadfin3 --exp O36 \
    --num_steps 10 --output /content/drive/MyDrive/vilio/export/O/attention --extract_type attention \
    --conf_modality img --conf_unit_size 2 --preds_threshold .0

### gradient
python hm.py --seed 42 --model O \
    --test dev_seen --lr 1e-5 --batchSize 8 --tr bert-large-uncased --epochs 5 --tsv \
    --num_features 36 --loadfin $loadfin3 --exp O36 \
    --num_steps 10 --output /content/drive/MyDrive/vilio/export/O/gradient --extract_type gradient \
    --conf_modality txt --conf_unit_size 2 --preds_threshold .0

python hm.py --seed 42 --model O \
    --test dev_seen --lr 1e-5 --batchSize 8 --tr bert-large-uncased --epochs 5 --tsv \
    --num_features 36 --loadfin $loadfin3 --exp O36 \
    --num_steps 10 --output /content/drive/MyDrive/vilio/export/O/gradient --extract_type gradient \
    --conf_modality img --conf_unit_size 2 --preds_threshold .0

# OSA
## Simple Average O-Model outputs
python utils/ens.py --enspath ./data/ --enstype sa --exp O365050

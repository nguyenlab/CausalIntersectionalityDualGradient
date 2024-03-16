#!/bin/bash

# Loading finetuned without having to move it to ./data
loadfin=${1:-./data/LASTtrain.pth}
loadfin2=${2:-./data/LASTtraindev.pth}
loadfin3=${2:-./data/LASTtraindev.pth}

# hm_VLMDB
python fts_lmdb/lmdb_conversion.py

# hm_V45
## 100 Feats, Seed 45
cp ./data/hm_vgattr7272.tsv ./data/HM_img.tsv

### attattr
python hm.py --seed 45 --model V \
    --test dev_seen --lr 1e-5 --batchSize 8 --tr bert-base-uncased --epochs 5 --reg \
    --num_features 100 --loadfin $loadfin --exp V45 \
    --output /content/drive/MyDrive/vilio/export/V/attattr --extract_type attattr \
    --conf_modality txt --conf_unit_size 2 --preds_threshold .0 --highlight_sample

python hm.py --seed 45 --model V \
    --test dev_seen --lr 1e-5 --batchSize 8 --tr bert-base-uncased --epochs 5 --reg \
    --num_features 100 --loadfin $loadfin --exp V45 \
    --output /content/drive/MyDrive/vilio/export/V/attattr --extract_type attattr \
    --conf_modality img --conf_unit_size 2 --preds_threshold .0 --highlight_sample

### attention
python hm.py --seed 45 --model V \
    --test dev_seen --lr 1e-5 --batchSize 8 --tr bert-base-uncased --epochs 5 --reg \
    --num_features 100 --loadfin $loadfin --exp V45 \
    --output /content/drive/MyDrive/vilio/export/V/attention --extract_type attention \
    --conf_modality txt --conf_unit_size 2 --preds_threshold .0 

python hm.py --seed 45 --model V \
    --test dev_seen --lr 1e-5 --batchSize 8 --tr bert-base-uncased --epochs 5 --reg \
    --num_features 100 --loadfin $loadfin --exp V45 \
    --output /content/drive/MyDrive/vilio/export/V/attention --extract_type attention \
    --conf_modality img --conf_unit_size 2 --preds_threshold .0 

### gradient
python hm.py --seed 45 --model V \
    --test dev_seen --lr 1e-5 --batchSize 8 --tr bert-base-uncased --epochs 5 --reg \
    --num_features 100 --loadfin $loadfin --exp V45 \
    --output /content/drive/MyDrive/vilio/export/V/gradient --extract_type gradient \
    --conf_modality txt --conf_unit_size 2 --preds_threshold .0 

python hm.py --seed 45 --model V \
    --test dev_seen --lr 1e-5 --batchSize 8 --tr bert-base-uncased --epochs 5 --reg \
    --num_features 100 --loadfin $loadfin --exp V45 \
    --output /content/drive/MyDrive/vilio/export/V/gradient --extract_type gradient \
    --conf_modality img --conf_unit_size 2 --preds_threshold .0 

### text-only
python hm.py --seed 45 --model V \
    --test dev_seen --lr 1e-5 --batchSize 8 --tr bert-base-uncased --epochs 5 --reg \
    --num_features 100 --loadfin $loadfin --exp V45 \
    --output /content/drive/MyDrive/vilio/export/V/attattr/text_only --extract_type attattr \
    --conf_modality txt --conf_unit_size 2 --preds_threshold .0 \
    --text_only

python hm.py --seed 45 --model V \
    --test dev_seen --lr 1e-5 --batchSize 8 --tr bert-base-uncased --epochs 5 --reg \
    --num_features 100 --loadfin $loadfin --exp V45 \
    --output /content/drive/MyDrive/vilio/export/V/attattr/text_only --extract_type attattr \
    --conf_modality img --conf_unit_size 2 --preds_threshold .0 \
    --text_only

# hm_V90
## 100 Feats, Seed 90
cp ./data/hm_vgattr5050.tsv ./data/HM_img.tsv

### attattr
python hm.py --seed 90 --model V \
    --test dev_seen --lr 1e-5 --batchSize 8 --tr bert-base-uncased --epochs 5 --reg \
    --num_features 100 --loadfin $loadfin --exp V90 \
    --output /content/drive/MyDrive/vilio/export/V/attattr --extract_type attattr \
    --conf_modality txt --conf_unit_size 2 --preds_threshold .0 --highlight_sample

python hm.py --seed 90 --model V \
    --test dev_seen --lr 1e-5 --batchSize 8 --tr bert-base-uncased --epochs 5 --reg \
    --num_features 100 --loadfin $loadfin --exp V90 \
    --output /content/drive/MyDrive/vilio/export/V/attattr --extract_type attattr \
    --conf_modality img --conf_unit_size 2 --preds_threshold .0 --highlight_sample

### attention
python hm.py --seed 90 --model V \
    --test dev_seen --lr 1e-5 --batchSize 8 --tr bert-base-uncased --epochs 5 --reg \
    --num_features 100 --loadfin $loadfin --exp V90 \
    --output /content/drive/MyDrive/vilio/export/V/attention --extract_type attention \
    --conf_modality txt --conf_unit_size 2 --preds_threshold .0 

python hm.py --seed 90 --model V \
    --test dev_seen --lr 1e-5 --batchSize 8 --tr bert-base-uncased --epochs 5 --reg \
    --num_features 100 --loadfin $loadfin --exp V90 \
    --output /content/drive/MyDrive/vilio/export/V/attention --extract_type attention \
    --conf_modality img --conf_unit_size 2 --preds_threshold .0 

### gradient
python hm.py --seed 90 --model V \
    --test dev_seen --lr 1e-5 --batchSize 8 --tr bert-base-uncased --epochs 5 --reg \
    --num_features 100 --loadfin $loadfin --exp V90 \
    --output /content/drive/MyDrive/vilio/export/V/gradient --extract_type gradient \
    --conf_modality txt --conf_unit_size 2 --preds_threshold .0 

python hm.py --seed 90 --model V \
    --test dev_seen --lr 1e-5 --batchSize 8 --tr bert-base-uncased --epochs 5 --reg \
    --num_features 100 --loadfin $loadfin --exp V90 \
    --output /content/drive/MyDrive/vilio/export/V/gradient --extract_type gradient \
    --conf_modality img --conf_unit_size 2 --preds_threshold .0 

### text-only
python hm.py --seed 90 --model V \
    --test dev_seen --lr 1e-5 --batchSize 8 --tr bert-base-uncased --epochs 5 --reg \
    --num_features 100 --loadfin $loadfin --exp V90 \
    --output /content/drive/MyDrive/vilio/export/V/attattr/text_only --extract_type attattr \
    --conf_modality txt --conf_unit_size 2 --preds_threshold .0 \
    --text_only

python hm.py --seed 90 --model V \
    --test dev_seen --lr 1e-5 --batchSize 8 --tr bert-base-uncased --epochs 5 --reg \
    --num_features 100 --loadfin $loadfin --exp V90 \
    --output /content/drive/MyDrive/vilio/export/V/attattr/text_only --extract_type attattr \
    --conf_modality img --conf_unit_size 2 --preds_threshold .0 \
    --text_only

# hm_V135
## 100 Feats, Seed 135
cp ./data/hm_vgattr3636.tsv ./data/HM_img.tsv

### attattr
python hm.py --seed 135 --model V \
    --test dev_seen --lr 1e-5 --batchSize 8 --tr bert-base-uncased --epochs 5 --reg \
    --num_features 100 --loadfin $loadfin --exp V135 \
    --output /content/drive/MyDrive/vilio/export/V/attattr --extract_type attattr \
    --conf_modality txt --conf_unit_size 2 --preds_threshold .0 --highlight_sample

python hm.py --seed 135 --model V \
    --test dev_seen --lr 1e-5 --batchSize 8 --tr bert-base-uncased --epochs 5 --reg \
    --num_features 100 --loadfin $loadfin --exp V135 \
    --output /content/drive/MyDrive/vilio/export/V/attattr --extract_type attattr \
    --conf_modality img --conf_unit_size 2 --preds_threshold .0 --highlight_sample

### attention
python hm.py --seed 135 --model V \
    --test dev_seen --lr 1e-5 --batchSize 8 --tr bert-base-uncased --epochs 5 --reg \
    --num_features 100 --loadfin $loadfin --exp V135 \
    --output /content/drive/MyDrive/vilio/export/V/attention --extract_type attention \
    --conf_modality txt --conf_unit_size 2 --preds_threshold .0 

python hm.py --seed 135 --model V \
    --test dev_seen --lr 1e-5 --batchSize 8 --tr bert-base-uncased --epochs 5 --reg \
    --num_features 100 --loadfin $loadfin --exp V135 \
    --output /content/drive/MyDrive/vilio/export/V/attention --extract_type attention \
    --conf_modality img --conf_unit_size 2 --preds_threshold .0 

### gradient
python hm.py --seed 135 --model V \
    --test dev_seen --lr 1e-5 --batchSize 8 --tr bert-base-uncased --epochs 5 --reg \
    --num_features 100 --loadfin $loadfin --exp V135 \
    --output /content/drive/MyDrive/vilio/export/V/gradient --extract_type gradient \
    --conf_modality txt --conf_unit_size 2 --preds_threshold .0 

python hm.py --seed 135 --model V \
    --test dev_seen --lr 1e-5 --batchSize 8 --tr bert-base-uncased --epochs 5 --reg \
    --num_features 100 --loadfin $loadfin --exp V135 \
    --output /content/drive/MyDrive/vilio/export/V/gradient --extract_type gradient \
    --conf_modality img --conf_unit_size 2 --preds_threshold .0 

### text-only
python hm.py --seed 135 --model V \
    --test dev_seen --lr 1e-5 --batchSize 8 --tr bert-base-uncased --epochs 5 --reg \
    --num_features 100 --loadfin $loadfin --exp V135 \
    --output /content/drive/MyDrive/vilio/export/V/attattr/text_only --extract_type attattr \
    --conf_modality txt --conf_unit_size 2 --preds_threshold .0 \
    --text_only

python hm.py --seed 135 --model V \
    --test dev_seen --lr 1e-5 --batchSize 8 --tr bert-base-uncased --epochs 5 --reg \
    --num_features 100 --loadfin $loadfin --exp V135 \
    --output /content/drive/MyDrive/vilio/export/V/attattr/text_only --extract_type attattr \
    --conf_modality img --conf_unit_size 2 --preds_threshold .0 \
    --text_only

# hm_VSA
## Simple Average
python utils/ens.py --enspath ./data/ --enstype sa --exp VLMDB
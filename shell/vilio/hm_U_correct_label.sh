#!/bin/bash

# Loading finetuned without having to move it to ./data
loadfin=${1:-./data/LASTtrain.pth}
loadfin2=${2:-./data/LASTtraindev.pth}
loadfin3=${2:-./data/LASTtraindev.pth}

# hm_U72
## 72 Feats, Seed 86
cp ./data/hm_vgattr7272.tsv ./data/HM_img.tsv

### attattr
python hm.py --seed 86 --model U \
    --test dev_seen --lr 1e-5 --batchSize 8 --tr bert-large-cased --epochs 5 --tsv \
    --num_features 72 --num_pos 6 --loadfin $loadfin --exp U72 \
    --output /content/drive/MyDrive/vilio/export/correct_label/U/attattr --extract_type attattr \
    --conf_modality txt --conf_unit_size 2 --preds_threshold .0 --pick_correct_labels 

python hm.py --seed 86 --model U \
    --test dev_seen --lr 1e-5 --batchSize 8 --tr bert-large-cased --epochs 5 --tsv \
    --num_features 72 --num_pos 6 --loadfin $loadfin --exp U72 \
    --output /content/drive/MyDrive/vilio/export/correct_label/U/attattr --extract_type attattr \
    --conf_modality img --conf_unit_size 2 --preds_threshold .0 --pick_correct_labels 

### attention
python hm.py --seed 86 --model U \
    --test dev_seen --lr 1e-5 --batchSize 8 --tr bert-large-cased --epochs 5 --tsv \
    --num_features 72 --num_pos 6 --loadfin $loadfin --exp U72 \
    --output /content/drive/MyDrive/vilio/export/correct_label/U/attention --extract_type attention \
    --conf_modality txt --conf_unit_size 2 --preds_threshold .0 --pick_correct_labels 

python hm.py --seed 86 --model U \
    --test dev_seen --lr 1e-5 --batchSize 8 --tr bert-large-cased --epochs 5 --tsv \
    --num_features 72 --num_pos 6 --loadfin $loadfin --exp U72 \
    --output /content/drive/MyDrive/vilio/export/correct_label/U/attention --extract_type attention \
    --conf_modality img --conf_unit_size 2 --preds_threshold .0 --pick_correct_labels 

### gradient
python hm.py --seed 86 --model U \
    --test dev_seen --lr 1e-5 --batchSize 8 --tr bert-large-cased --epochs 5 --tsv \
    --num_features 72 --num_pos 6 --loadfin $loadfin --exp U72 \
    --output /content/drive/MyDrive/vilio/export/correct_label/U/gradient --extract_type gradient \
    --conf_modality txt --conf_unit_size 2 --preds_threshold .0 --pick_correct_labels 

python hm.py --seed 86 --model U \
    --test dev_seen --lr 1e-5 --batchSize 8 --tr bert-large-cased --epochs 5 --tsv \
    --num_features 72 --num_pos 6 --loadfin $loadfin --exp U72 \
    --output /content/drive/MyDrive/vilio/export/correct_label/U/gradient --extract_type gradient \
    --conf_modality img --conf_unit_size 2 --preds_threshold .0 --pick_correct_labels 

# hm_U50
## 50 Feats, Seed 43
cp ./data/hm_vgattr5050.tsv ./data/HM_img.tsv

### attattr
python hm.py --seed 43 --model U \
    --test dev_seen --lr 1e-5 --batchSize 8 --tr bert-large-cased --epochs 5 --tsv \
    --num_features 50 --num_pos 6 --loadfin $loadfin2 --exp U50 \
    --output /content/drive/MyDrive/vilio/export/correct_label/U/attattr --extract_type attattr \
    --conf_modality txt --conf_unit_size 2 --preds_threshold .0 --pick_correct_labels 

python hm.py --seed 43 --model U \
    --test dev_seen --lr 1e-5 --batchSize 8 --tr bert-large-cased --epochs 5 --tsv \
    --num_features 50 --num_pos 6 --loadfin $loadfin2 --exp U50 \
    --output /content/drive/MyDrive/vilio/export/correct_label/U/attattr --extract_type attattr \
    --conf_modality img --conf_unit_size 2 --preds_threshold .0 --pick_correct_labels 

### attention
python hm.py --seed 43 --model U \
    --test dev_seen --lr 1e-5 --batchSize 8 --tr bert-large-cased --epochs 5 --tsv \
    --num_features 50 --num_pos 6 --loadfin $loadfin2 --exp U50 \
    --output /content/drive/MyDrive/vilio/export/correct_label/U/attention --extract_type attention \
    --conf_modality txt --conf_unit_size 2 --preds_threshold .0 --pick_correct_labels 

python hm.py --seed 43 --model U \
    --test dev_seen --lr 1e-5 --batchSize 8 --tr bert-large-cased --epochs 5 --tsv \
    --num_features 50 --num_pos 6 --loadfin $loadfin2 --exp U50 \
    --output /content/drive/MyDrive/vilio/export/correct_label/U/attention --extract_type attention \
    --conf_modality img --conf_unit_size 2 --preds_threshold .0 --pick_correct_labels 

### gradient
python hm.py --seed 43 --model U \
    --test dev_seen --lr 1e-5 --batchSize 8 --tr bert-large-cased --epochs 5 --tsv \
    --num_features 50 --num_pos 6 --loadfin $loadfin2 --exp U50 \
    --output /content/drive/MyDrive/vilio/export/correct_label/U/gradient --extract_type gradient \
    --conf_modality txt --conf_unit_size 2 --preds_threshold .0 --pick_correct_labels 

python hm.py --seed 43 --model U \
    --test dev_seen --lr 1e-5 --batchSize 8 --tr bert-large-cased --epochs 5 --tsv \
    --num_features 50 --num_pos 6 --loadfin $loadfin2 --exp U50 \
    --output /content/drive/MyDrive/vilio/export/correct_label/U/gradient --extract_type gradient \
    --conf_modality img --conf_unit_size 2 --preds_threshold .0 --pick_correct_labels 

# hm_U36
## 36 Feats, Seed 129
cp ./data/hm_vgattr3636.tsv ./data/HM_img.tsv

### attattr
python hm.py --seed 129 --model U \
    --test dev_seen --lr 1e-5 --batchSize 8 --tr bert-large-cased --epochs 5 --tsv \
    --num_features 36 --num_pos 6 --loadfin $loadfin3 --exp U36 \
    --output /content/drive/MyDrive/vilio/export/correct_label/U/attattr --extract_type attattr \
    --conf_modality txt --conf_unit_size 2 --preds_threshold .0 --pick_correct_labels 

python hm.py --seed 129 --model U \
    --test dev_seen --lr 1e-5 --batchSize 8 --tr bert-large-cased --epochs 5 --tsv \
    --num_features 36 --num_pos 6 --loadfin $loadfin3 --exp U36 \
    --output /content/drive/MyDrive/vilio/export/correct_label/U/attattr --extract_type attattr \
    --conf_modality img --conf_unit_size 2 --preds_threshold .0 --pick_correct_labels 

### attention
python hm.py --seed 129 --model U \
    --test dev_seen --lr 1e-5 --batchSize 8 --tr bert-large-cased --epochs 5 --tsv \
    --num_features 36 --num_pos 6 --loadfin $loadfin3 --exp U36 \
    --output /content/drive/MyDrive/vilio/export/correct_label/U/attention --extract_type attention \
    --conf_modality txt --conf_unit_size 2 --preds_threshold .0 --pick_correct_labels 

python hm.py --seed 129 --model U \
    --test dev_seen --lr 1e-5 --batchSize 8 --tr bert-large-cased --epochs 5 --tsv \
    --num_features 36 --num_pos 6 --loadfin $loadfin3 --exp U36 \
    --output /content/drive/MyDrive/vilio/export/correct_label/U/attention --extract_type attention \
    --conf_modality img --conf_unit_size 2 --preds_threshold .0 --pick_correct_labels 

### gradient
python hm.py --seed 129 --model U \
    --test dev_seen --lr 1e-5 --batchSize 8 --tr bert-large-cased --epochs 5 --tsv \
    --num_features 36 --num_pos 6 --loadfin $loadfin3 --exp U36 \
    --output /content/drive/MyDrive/vilio/export/correct_label/U/gradient --extract_type gradient \
    --conf_modality txt --conf_unit_size 2 --preds_threshold .0 --pick_correct_labels 

python hm.py --seed 129 --model U \
    --test dev_seen --lr 1e-5 --batchSize 8 --tr bert-large-cased --epochs 5 --tsv \
    --num_features 36 --num_pos 6 --loadfin $loadfin3 --exp U36 \
    --output /content/drive/MyDrive/vilio/export/correct_label/U/gradient --extract_type gradient \
    --conf_modality img --conf_unit_size 2 --preds_threshold .0 --pick_correct_labels 


# hm_USA
## Simple Average
python utils/ens.py --enspath ./data/ --enstype sa --exp U365072
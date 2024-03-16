#!/bin/sh

# overwrite vilio scripts for attribute scoring
## common
cd /MemesModalityEvaluation-2540/script/vilio; cp hm_data.py /vilio/fts_lmdb
cd /MemesModalityEvaluation-2540/script/vilio; cp hm_data_tsv.py /vilio/fts_tsv
## E
cd /MemesModalityEvaluation-2540/script/vilio; cp ernie_vil_gradient.py /vilio/ernie-vil/model/ernie_vil.py
# cd /MemesModalityEvaluation-2540/script/vilio; cp vl_transformer_encoder_gradient.py /vilio/ernie-vil/model
cd /MemesModalityEvaluation-2540/script/vilio; cp finetune_gradient.py /vilio/ernie-vil/finetune.py
## O
cd /MemesModalityEvaluation-2540/script/vilio; cp modeling_bertO_gradient.py /vilio/src/vilio
cd /MemesModalityEvaluation-2540/script/vilio; cp entryO_gradient.py /vilio/entryO.py
cd /MemesModalityEvaluation-2540/script/vilio; cp hm_gradient.py /vilio/hm.py
cd /MemesModalityEvaluation-2540/script/vilio; cp param_gradient.py /vilio/param.py
cd /MemesModalityEvaluation-2540/shell/vilio; cp hm_O.sh /vilio/bash/inference/O
cd /MemesModalityEvaluation-2540/shell/vilio; cp hm_O_correct_label.sh /vilio/bash/inference/O
## U
cd /MemesModalityEvaluation-2540/script/vilio; cp modeling_bertU_gradient.py /vilio/src/vilio
cd /MemesModalityEvaluation-2540/script/vilio; cp entryU_gradient.py /vilio/entryU.py
cd /MemesModalityEvaluation-2540/shell/vilio; cp hm_U.sh /vilio/bash/inference/U
cd /MemesModalityEvaluation-2540/shell/vilio; cp hm_U_correct_label.sh /vilio/bash/inference/U
## V
cd /MemesModalityEvaluation-2540/script/vilio; cp modeling_bertV_gradient.py /vilio/src/vilio
cd /MemesModalityEvaluation-2540/script/vilio; cp entryV_gradient.py /vilio/entryV.py
cd /MemesModalityEvaluation-2540/shell/vilio; cp hm_V.sh /vilio/bash/inference/V
cd /MemesModalityEvaluation-2540/shell/vilio; cp hm_V_correct_label.sh /vilio/bash/inference/V
cd /MemesModalityEvaluation-2540/script/vilio; cp modeling_bert_gradient.py /vilio/src/vilio/transformers

# added
mkdir /vilio/modality_evaluation; mkdir /vilio/attattr
cp /MemesModalityEvaluation-2540/script/attattr/*  /vilio/attattr
cp /MemesModalityEvaluation-2540/script/modality_evaluation/* /vilio/modality_evaluation
cd /MemesModalityEvaluation-2540/shell; cp micace_evaluator.sh /vilio/bash
cp /MemesModalityEvaluation-2540/script/modality_evaluation/micace_evaluator.py /vilio/bash

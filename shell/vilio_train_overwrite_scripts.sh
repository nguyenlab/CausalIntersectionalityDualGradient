#!/bin/sh

# overwrite vilio scripts for training
## common
cd /MemesModalityEvaluation-2540/script/vilio/train; cp pandas_scripts.py /vilio/utils; cp hm_data.py /vilio/fts_lmdb; cp hm.py /vilio

## vlmo
mkdir /vilio/vlmo
cd /MemesModalityEvaluation-2540/script/vlmo; cp entry_VLMo.py /vilio;rm entry_VLMo.py; cp pretrain_VLMo.py /vilio;rm pretrain_VLMo.py
cp /MemesModalityEvaluation-2540/script/vlmo/* /vilio/vlmo
#!/bin/sh

TF_DIR="/usr/local/lib/python3.10/dist-packages/transformers"
SCRIPT_DIR="/content/MemesModalityEvaluation-2540/script/llama/hf"

# overwrite hf scripts for llama2
## Transformers
cp $SCRIPT_DIR/text_generation.py $TF_DIR/pipelines/text_generation.py 
cp $SCRIPT_DIR/modeling_llama.py $TF_DIR/models/llama/modeling_llama.py 

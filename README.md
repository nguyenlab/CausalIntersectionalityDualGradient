# CausalIntersectionalityDualGradient
## Summary
This repository contains the codes for the paper [Causal Intersectionality and Dual Form of Gradient Descent for Multimodal Analysis: A Case Study on Hateful Memes](https://arxiv.org/abs/2308.11585).

# How to Reproduce the Result
## Summary
| Result | How to Reproduce |
| --- | --- |
| Figure 3-5 | miATE / MIDAS |
| Table 5 | miATE vs MIDAS |
| Table 6,7 | meta-optimization |

## Preparation
- Use Google Drive to place data/token.txt on MyDrive/vilio
- For LLM experiment, clone this repo and `pip install -r llm_requirements.txt`
## miATE / MIDAS
- Run `notebook/vilio_gradient.ipynb` 
## miATE vs MIDAS
- Run `notebook/intersectioanlity_tables.ipynb`

## LLM
### BLIP-2 image captioning
- Run `script/image2text.py`
```
python blip2/image2text.py \
    --input_dir <path-to-Hateful-Memes> \
    --output_dir <path-to-BLIP2-Captions>
```
### BLIP-2 and LLaMA
- Run `notebook/hf_llama2.ipynb`
### meta-optimization
- Run `notebook/intersectioanlity_tables.ipynb`
# Appendix
## BLIP-2 and BERT
- Run `script/bert_midas.py`
```
python blip2/bert_midas.py \
    --memes_dir <path-to-Hateful-Memes> \
    --caption_dir <path-to-BLIP2-Captions> \
    --output_dir <path-to-output> \
    --exp_name test --eval_set test_seen --train_set train,dev_seen \
    --random_seed <seeds from 1987 to 1991>
```

# Citation
```
@misc{2308.11585,
    Author = {Yosuke Miyanishi and Minh Le Nguyen},
    Title = {Causal Intersectionality and Dual Form of Gradient Descent for Multimodal Analysis: a Case Study on Hateful Memes},
    Year = {2023},
    Eprint = {arXiv:2308.11585},
}
```
# STOCHCA: A NOVEL APPROACH FOR EXPLOITING PRETRAINED MODELS WITH CROSS-ATTENTION

### Introduction
StochCA (Stochastic Cross-Attention) introduces a fine-tuning method for Transformer architectures, aimed at enhancing the utilization of large-scale pretrained models across various target tasks. Unlike traditional fine-tuning, which may not fully leverage the knowledge embedded in pretrained models, StochCA employs a novel strategy that modifies the self-attention mechanism to incorporate cross-attention stochastically. This method allows for the selective exploitation of knowledge from pretrained models, significantly improving performance in transfer learning and domain generalization tasks.

### Requirements
- python == 3.9.12
- torch == 1.12.0
- torchvision == 0.13.0
- numpy == 1.22.0
- Pillow

### Results
- Log files will be saved as follows :
```
|---- domain_generalization
|     |---- <output_dir>
|           |---- err.txt
|           |---- out.txt
|           |---- results.json
...
|---- transfer_learning
|     |---- logs/
|           |---- CUB
|                 |---- StochCA
|                       |---- <word_dir>
|                             |---- config.json
|                             |---- train_log.json
```
### Acknowlegdement

This code is built on [DomainBed](https://github.com/kakaobrain/miro/tree/main) and [DoPrompt](https://github.com/zhengzangw/DoPrompt/tree/main). We appreciate to their authors for sharing the code.

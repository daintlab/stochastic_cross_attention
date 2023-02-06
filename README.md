# stochastic_cross_attention

## Prepare dataset
- Download dataset and run ```preprocess.py``` (modify dataset and data_dir line)
- Official test/train split will be copied to ```test/```, ```train_all/``` under ```data_dir```
- Official train split will be separated into ```train/``` and ```val/``` with ratio of 9 : 1
- Processed datasets are in ```/data/transfer_benchmarks``` for F,G,I Server

## Usage
- Train for validation & hyperparameter selection
```
CUDA_VISIBLE_DEVICES=0 python train.py --method <algorithm> --use_val \
--data_dir <path to data dir> --dataset <dataset> --work_dir <path to result dir>
```
- Train with selected hyperparameter(Use 50% of train data)
```
CUDA_VISIBLE_DEVICES=0 python train.py --method <algorithm> --ratio 0.5 \
--data_dir <path to data dir> --dataset <dataset> --work_dir <path to result dir>
```
- Available algorithms
  - ```ERM``` : naive fine-tuning
  - ```L2SP``` : L2-SP
  - ```BSS``` : Batch spectral shrinkage
  - ```CoTuning``` : Co-tuning
  - ```StochCA``` : stochastic cross attention
  - ```CoTuningStochCA``` : Co-tuning + stochastic cross attention
- train log and config file will be saved in ```./log/<dataset>/<algorithm>/<work_dir> ```

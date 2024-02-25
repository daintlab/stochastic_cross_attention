# StochCA in Domain Generalization

## Prepare Datasets
```python -m domainbed.scripts.download --data_dir <path to save dataset>```

## Usage
```
CUDA_VISIBLE_DEVICES=0 python -m scripts.train --data_dir <path to data dir> --steps 5001 --dataset <dataset> \
--test_env <target domain> --algorithm StochCA --output_dir <path to output dir> \
--hparams '{"lr": 1e-5, "lr_classifier": 1e-5, "ca_prob": 0.1, "weight_decay": 1e-2, "resnet_dropout": 0}' --seed <seed> --checkpoint_freq 200
```

- Avalable datasets
  - ```PACS```
  - ```OfficeHome```
  - ```VLCS```
  - ```DomainNet```

- Available algorithms
  - ```ERM```
  - ```StochCA```
  - ```StochCA_CLIP```
  - ```DoPrompt```
  - ```...```
       
- train log and config file will be saved in ``` ./<output_dir> ```

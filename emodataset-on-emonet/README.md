# Fine-tuning EmoNet on Emodataset

This repo contains the scripts used for fine-tuning [EmoNet](https://github.com/face-analysis/emonet) on a dataset called [Emodataset](https://www.kaggle.com/datasets/susmitdas1053/emodataset?select=Emodataset). It also contains jupyter notebooks used for testing, analysis, and graphing/visualizations. 

If intending to run, download emodataset from kaggle (link above) and pass dataset path to scripts. 

By Matt Stirling as part of Affective Computing (2025), University of Oulu. 


## Setup

I used python version 3.10.

### 1. Create dedicated environment

I used venv, but conda should work. But I installed all packages with pip. 

### 2. Install PyTorch:
Run:
```
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
```
or follow [instructions](https://pytorch.org/#:~:text=and%20easy%20scaling.-,Install%20PyTorch,of%20PyTorch.%20Note%20that%20LibTorch%20is%20only%20available%20for%20C%2B%2B.,-NOTE%3A%20Latest) on official PyTorch website.

### 3. Install requirements.txt
```
pip install -r requirements.txt
```

## Running

The scripts for fine-tuning and evaluation exist in `scripts/`, and in order to run them it is recommended to run them as modules. E.G.:

```
python3 -m scripts.train_emonet [ARGS]
```

Run script with `-h` for list of options. Important ones include `--pretrained_params` and `--dataset_root`. 

Below the scripts are described further:

| SCRIPT | DESCRIPTION |
| --- | --- |
| `scripts/train_emonet.py` | Script for fine tuning EmoNet on dataset with compatible labels.  |
| `scripts/evaluate_emonet.py` | Script for evaluating EmoNet given dataset and pretrained params.  |


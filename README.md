![](logo.svg)
___
**This repository has moved to [https://github.com/rgklab/detectron](https://github.com/rgklab/detectron)**

## Setup

### Environment

We recommend using conda environment to run `deeptst`. This can be done with the following commands:

```shell
# create and activate conda environment using a python version >= 3.9
conda create -n deeptst python=3.9
conda activate deeptst

# install the latest version of pytorch (tested for >= 1.9.0)
conda install pytorch torchvision cudatoolkit=11.3 -c pytorch

# install additional dependencies with pip
pip install -r requirements.txt
```

To run code in this repository make sure the root directory is in your python path. This can be done with the following
command:

```shell
export PYTHONPATH=$PYTHONPATH:/path/to/deeptst
```

### Datasets

We provide a simple config system to store dataset path mappings in the file `deeptst/config.yml`

```yaml
datasets:
  default: /datasets
  cifar10_1: /datasets/cifar-10-1
  uci_heart: /datasets/UCI
  camelyon17: /datasets/camelyon17
```

for more information see `deeptst/data/sample_data/README.md`.

### Pretrained Models

This repository requires several pretrained classifiers for running statistical tests.
In the coming weeks we will make pretrained models available for simple download.

### Running Detectron

There is work in progress to package Detectron in a robust and easy to deploy system.
For now, we all the code for our experiments in located in the `scratch/detectron` directory.

```shell
# run the cifar experiment using the standard config
# use python scratch/detectron/detectron_cifar.py --help for a documented list of options
❯ python scratch/detectron/detectron_cifar.py --run_name detectron_cifar
```

### Evaluating Detectron

The scratch files will write the output for each seed to a `.pt` file in a directory named `results/<run_name>`.

The script in `scratch/detectron/analysis.py` will read these files and produce a summary of the results for each test
described in the paper.

```shell
❯ python scratch/detectron/analysis.py --run_name detectron_cifar
# Output
→ 600 runs loaded
→ Running Disagreement Test
N = 10, 20, 50
TPR: .37 ± .05 AUC: 0.799 | TPR: .54 ± .05 AUC: 0.902 | TPR: .83 ± .04 AUC: 0.981
→ Running Entropy Test
N = 10, 20, 50
TPR: .35 ± .05 AUC: 0.712 | TPR: .56 ± .05 AUC: 0.866 | TPR: .92 ± .03 AUC: 0.981

```

![](https://github.com/tomginsberg/deeptst/blob/main/logo.svg)
___
*Official implementation of the Detectron Two Sample Test for High Dimensional Covariate Shift*

## Recommended Setup

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
cd deeptst
export PYTHONPATH=$PYTHONPATH:$(pwd)
```

### Setting up datasets

We provide a simple config system to store dataset path mappings in the file `deeptst/config.yml`

```yaml
datasets:
    default: /datasets
    cifar10_1: /datasets/cifar-10-1
    uci_heart: /datasets/UCI
    camelyon17: /datasets/camelyon17
```

for more information see `deeptst/data/sample_data/README.md`.

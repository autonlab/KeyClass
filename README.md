<!--- # KeyClass
Classifying Unstructured Clinical Notes via Automatic Weak Supervision --->

# TODOs:
- [ ] Make a quick logo
- [ ] Complete Readme
- [ ] Make a base class for the encoders. The custom encoder class is iffy now
- [ ] Add __str__ methods for each of the classes

----

`KeyClass` is a data-centric AI package for text classification withe label-names (and descriptions) only.


# Get started with tutorials
  - Run this [script](https://github.com/autonlab/KeyClass/blob/main/scripts/get_data.sh) to download the datasets
  - [TUTORIAL: simple example on dbpedia (can be easily changed)](https://github.com/autonlab/KeyClass/blob/main/scripts/example_train.ipynb)


----

# Installation

Python 3.8+ is  supported. Results were originally ran on Linux. 

Setup the environment with the following steps: 

``` bash
$ conda create -n keyclass python=3.8
$ conda activate keyclass
$ conda install -c pytorch pytorch=1.10.0 cudatoolkit=10.2
$ conda install -c conda-forge snorkel=0.9.8
$ conda install -c huggingface tokenizers=0.10.1
$ conda install -c huggingface transformers=4.11.3
$ conda install -c conda-forge sentence-transformers=2.0.0
$ conda install jupyter notebook
```
Alternatively, we have also provided the conda .ymp file, so the environment can be recreated using the following steps:
```
$ conda env create -f scripts/conda.yaml
$ conda activate keyclass
```

## Reproducing Results in [Classifying Unstructured Clinical Notes via Automatic Weak Supervision](https://arxiv.org/pdf/2206.12088.pdf)
```
$ cd scripts
$ bash get_data.sh
$ python run_all.py --config../config_files/config_imdb.yml
$ python run_all.py --config../config_files/config_agnews.yml
$ python run_all.py --config../config_files/config_dbpedia.yml
$ python run_all.py --config../config_files/config_amazon.yml
```

# Citation
```
@article{gao2022classifying,
  title={Classifying Unstructured Clinical Notes via Automatic Weak Supervision},
  author={Gao, Chufan and Goswami, Mononito and Chen, Jieshi and Dubrawski, Artur},
  journal={Machine Learning for Healthcare Conference},
  year={2022},
  organization={PMLR}
}
```

Copyright (c) 2022 Carnegie Mellon University, Auton Lab.

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

See [MIT LICENSE](https://github.com/autonlab/KeyClass/blob/main/LICENSE) for details.


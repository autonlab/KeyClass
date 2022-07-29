# KeyClass: Text Classification with Label-Descriptions Only

`KeyClass` is a general weakly-supervised text classification framework that learns from *class-label descriptions only*, without the need to use any human-labeled documents. It leverages the linguistic domain knowledge stored within pre-trained language models and the data programming framework to assign code labels to individual texts. We demonstrate the efficacy and flexibility of our method by comparing it to state-of-the-art weak text classifiers across four real-world text classification datasets.

Code for the paper [Classifying Unstructured Clinical Notes via Automatic Weak Supervision](https://arxiv.org/abs/2206.12088).

----

## Get started with tutorials
  - Run this [script](https://github.com/autonlab/KeyClass/blob/main/scripts/get_data.sh) to download the datasets
  - [TUTORIAL: simple example on dbpedia (can be easily changed)](https://github.com/autonlab/KeyClass/blob/main/scripts/example_train.ipynb)


----

## Installation

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
Additionally, we have released our [pretrained models](https://github.com/autonlab/KeyClass/releases/tag/v1.0). Please see the tutorial notebook above on evaluating trained models.

## Citation
If you use our code please cite the following paper. 
```
@article{gao2022classifying,
  title={Classifying Unstructured Clinical Notes via Automatic Weak Supervision},
  author={Gao, Chufan and Goswami, Mononito and Chen, Jieshi and Dubrawski, Artur},
  journal={Machine Learning for Healthcare Conference},
  year={2022},
  organization={PMLR}
}
```

## License

MIT License

Copyright (c) 2022 Carnegie Mellon University, Auton Lab.

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

See [MIT LICENSE](https://github.com/autonlab/KeyClass/blob/main/LICENSE) for details.

<img align="right" height ="120px" src="https://www.cs.cmu.edu/~chiragn/cmu_logo.jpeg">
<img align="right" height ="110px" src="https://www.cs.cmu.edu/~chiragn/auton_logo.png"> 

<h1 align="center">KeyClass: Text Classification with Label-Descriptions Only</h1>

<p align="center">
<img alt="License" src="https://img.shields.io/badge/License-MIT-yellow.svg">
<img alt="Visitors" src="https://visitor-badge.glitch.me/badge?page_id=autonlab/KeyClass">
</p>

`KeyClass` is a general weakly-supervised text classification framework that learns from *class-label descriptions only*, without the need to use any human-labeled documents. It leverages the linguistic domain knowledge stored within pre-trained language models and the data programming framework to assign labels to documents. We demonstrate the efficacy and flexibility of our method by comparing it to state-of-the-art weak text classifiers across four real-world text classification datasets.

## Contents

1. [Overview of Methodology](#methodology) 
2. [KeyClass Outperforms Advanced Weakly Supervised Models](#results) 
3. [Datasets](#datasets)
4. [Tutorial](#tutorial)
5. [Installation](#installation)
6. [Reproducing Results in Classifying Unstructured Clinical Notes via Automatic Weak Supervision](#reproduce)
7. [Citation](#citation)
8. [Contributing](#contrib)
9. [License](#license)

<a id="methodology"></a>
## Overview of Methodology 

<p align="center">
<img height ="300px" src="assets/KeyClass.png">
</p>

**Figure.1** From class descriptions only, KeyClass classifies documents without access to any labeled data. It automatically creates interpretable labeling functions (LFs) by extracting frequent keywords and phrases that are highly indicative of a particular class from the unlabeled text using a pre-trained language model. It then uses these LFs along with Data Programming (DP) to generate probabilistic labels for training data, which are used to train a downstream classifier [(Ratner et al., 2016)](https://arxiv.org/abs/1605.07723)

<a id="results"></a>
## `KeyClass` Outperforms Advanced Weakly Supervised Models

<p align="center">
<img height ="120px" src="assets/result_table.png">
</p>

**Table 1.    Classification Accuracy.** `KeyClass` outperforms state-of-the-art weakly supervised methods on 4 real-world text classification datasets. We report our model’s accuracy with a 95% bootstrap confidence intervals. Results for Dataless, WeSTClass,
LOTClass, and BERT are reported from [(Meng et al., 2020)](https://arxiv.org/abs/2010.07245).


----
<a id="datasets"></a>
## Datasets

To download the datasets used in the paper, run this [script](https://github.com/autonlab/KeyClass/blob/main/scripts/get_data.sh)

<a id="tutorial"></a>
## Tutorial
To familiarize yourself with `KeyClass`, please go through the following [tutorial](https://github.com/autonlab/KeyClass/blob/main/tutorials/Tutorial%20on%20IMDB.ipynb) which trains a text classifier from scratch on the DBpedia dataset.

<a id="installation"></a>
## Installation

All models were built and trained using PyTorch 1.8.1 using Python 3.8.1. Experiments were carried out on a computing cluster, with a typical machine having 40 Intel Xeon Silver 4210 CPUs, 187 GB of RAM, and 4 NVIDIA RTX2080 GPUs.

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
Alternatively, we have also provided the conda .yaml file, so the environment can be recreated using the following steps:
```
$ conda env create -f scripts/environment.yaml
$ conda activate keyclass
```

<a id="reproduce"></a>
## Reproducing Results in [Classifying Unstructured Clinical Notes via Automatic Weak Supervision](https://arxiv.org/pdf/2206.12088.pdf)
To reproduce the results in our paper, run the following commands. 
```
$ cd scripts
$ bash get_data.sh
$ python run_all.py --config../config_files/config_imdb.yml
$ python run_all.py --config../config_files/config_agnews.yml
$ python run_all.py --config../config_files/config_dbpedia.yml
$ python run_all.py --config../config_files/config_amazon.yml
```
Additionally, we release our [pretrained models](https://github.com/autonlab/KeyClass/releases/tag/v1.0). Please see the tutorial notebook above on evaluating trained models.

<a id="citation"></a>
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
<a id="contrib"></a>
## Contributing

`KeyClass` is [on GitHub]. We welcome bug reports and pull requests.

[on GitHub]: https://github.com/autonlab/KeyClass.git

<a id="license"></a>
## License

MIT License

Copyright (c) 2022 Carnegie Mellon University, Auton Lab.

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

See [MIT LICENSE](https://github.com/autonlab/KeyClass/blob/main/LICENSE) for details.

<img align="right" height ="120px" src="assets/cmu_logo.png">
<img align="right" height ="110px" src="assets/autonlab_logo.png"> 

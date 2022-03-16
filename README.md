<!--- # KeyClass
Classifying Unstructured Clinical Notes via Automatic Weak Supervision --->

TODOs:
- [ ] Make a quick logo
- [ ] Add tutorial(s)
- [ ] Add the 1-3 line code snippet
- [ ] Complete Readme
- [ ] 
<!--- ![](https://raw.githubusercontent.com/cleanlab/assets/master/cleanlab/cleanlab_logo.png)--->

`KeyClass` is a data-centric AI package for text classification withe label-names (and descriptions) only.

<!---`cleanlab` `clean`s `lab`els and supports **finding, quantifying, and learning** with label issues in datasets. See datasets cleaned with `cleanlab` at [labelerrors.com](https://labelerrors.com).

Check out the: [documentation](https://docs.cleanlab.ai/), [examples](https://github.com/cleanlab/examples), and [installation instructions](https://github.com/cleanlab/cleanlab#installation)

`cleanlab` is powered by **confident learning**, published in this [paper](https://jair.org/index.php/jair/article/view/12125) | [blog](https://l7.curtisnorthcutt.com/confident-learning).--->

<!---[![pypi](https://img.shields.io/pypi/v/cleanlab.svg)](https://pypi.org/pypi/cleanlab/) [![os](https://img.shields.io/badge/platform-windows%20%7C%20macos%20%7C%20linux-lightgrey)](https://pypi.org/pypi/cleanlab/) [![py\_versions](https://img.shields.io/badge/python-2.7%20%7C%203.6%2B-blue)](https://pypi.org/pypi/cleanlab/) [![build\_status](https://github.com/cleanlab/cleanlab/workflows/CI/badge.svg)](https://github.com/cleanlab/cleanlab/actions?query=workflow%3ACI) [![coverage](https://codecov.io/gh/cleanlab/cleanlab/branch/master/graph/badge.svg)](https://app.codecov.io/gh/cleanlab/cleanlab) [![docs](https://readthedocs.org/projects/cleanlab/badge/?version=latest)](https://docs.cleanlab.ai/)--->

# Get started with tutorials
  - (Easiest) Improve a simple classifier from 60% ---> 80% accuracy on the Iris dataset:
    - [TUTORIAL: simple cleanlab on Iris](https://github.com/cleanlab/examples/blob/master/iris_simple_example.ipynb)
<!---  - (Comprehensive) Image classification with noisy labels
    - [TUTORIAL: learning with noisy labels on CIFAR](https://github.com/cleanlab/examples/tree/master/cifar10)
  - Run Cleanlab on 4 datasets using 9 different classifiers/models:   
    - [TUTORIAL: classifier comparison](https://github.com/cleanlab/examples/blob/master/classifier_comparison.ipynb)
  - Find [label errors](https://arxiv.org/abs/2103.14749) in MNIST, ImageNet, CIFAR-10/100, Caltech-256, QuickDraw, Amazon Reviews, IMDB, 20 Newsgroups, AudioSet:
    - [TUTORIAL: Find Label Errors in the 10 most common ML benchmark test datasets with cleanlab](https://github.com/cleanlab/label-errors/blob/main/examples/Tutorial%20-%20How%20To%20Find%20Label%20Errors%20With%20CleanLab.ipynb)
  - Demystifying [Confident Learning](https://www.jair.org/index.php/jair/article/view/12125) (the theory and algorithms underlying `cleanlab`):
     - [TUTORIAL: confident learning with just numpy and for-loops](https://github.com/cleanlab/examples/blob/master/simplifying_confident_learning_tutorial.ipynb)
     - [TUTORIAL: visualizing confident learning](https://github.com/cleanlab/examples/blob/master/visualizing_confident_learning.ipynb)--->

-----
<!--- 
<details><summary><b>News! (2021) </b> -- <code>cleanlab</code> finds pervasive label errors in the most common ML test sets (<b>click to learn more</b>) </summary>
<p>
<ul>
<li> <b>Dec 2021 🎉</b>  NeurIPS published the <a href="https://arxiv.org/abs/2103.14749">label errors paper (Northcutt, Athalye, & Mueller, 2021)</a>.</li>
<li> <b>Apr 2021 🎉</b>  Journal of AI Research published the <a href="https://jair.org/index.php/jair/article/view/12125">confident learning paper (Northcutt, Jiang, & Chuang, 2021)</a>.</li>
<li> <b>Mar 2021 😲</b>  <code>cleanlab</code> used to find and fix label issues in 10 of the most common ML benchmark datasets, published in: <a href="https://datasets-benchmarks-proceedings.neurips.cc/paper/2021/hash/f2217062e9a397a1dca429e7d70bc6ca-Abstract-round1.html">NeurIPS 2021</a>. Along with <a href="https://arxiv.org/abs/2103.14749">the paper (Northcutt, Athalye, & Mueller, 2021)</a>, the authors launched <a href="https://labelerrors.com">labelerrors.com</a> where you can view the label issues in these datasets.</li>
</ul>
</p>
</details>

<details><summary><b>News! (2020) </b> -- <code>cleanlab</code> adds support for all OS, achieves state-of-the-art, supports co-teaching, and more (<b>click to learn more</b>) </summary>
<p>
<ul>
<li> <b>Dec 2020 🎉</b>  <code>cleanlab</code> supports NeurIPS workshop paper <a href="https://securedata.lol/camera_ready/28.pdf">(Northcutt, Athalye, & Lin, 2020)</a>.</li>
<li> <b>Dec 2020 🤖</b>  <code>cleanlab</code> supports <a href="https://github.com/cleanlab/cleanlab#pu-learning-with-cleanlab">PU learning</a>.</li>
<li> <b>Feb 2020 🤖</b>  <code>cleanlab</code> now natively supports Mac, Linux, and Windows.</li>
<li> <b>Feb 2020 🤖</b>  <code>cleanlab</code> now supports <a href="https://github.com/cleanlab/cleanlab/blob/master/cleanlab/coteaching.py">Co-Teaching</a> <a href="https://arxiv.org/abs/1804.06872">(Han et al., 2018)</a>.</li>
<li> <b>Jan 2020 🎉</b> <code>cleanlab</code> achieves state-of-the-art on CIFAR-10 with noisy labels. Code to reproduce:  <a href="https://github.com/cleanlab/examples/tree/master/cifar10">examples/cifar10</a>. This is a great place to see how to use cleanlab on real datasets (with predicted probabiliteis already precomputed for you).</li>
</ul>
</p>
</details>

Past release notes and **future features planned** is available [here](https://github.com/cleanlab/cleanlab/blob/master/cleanlab/version.py).

## So fresh, so `cleanlab`

`cleanlab` finds and cleans label issues in any dataset using [state-of-the-art algorithms](https://arxiv.org/abs/1911.00068) to find label issues, characterize noise, and learn in spite of it. `cleanlab` is fast: its built on optimized algorithms and parallelized across CPU threads automatically. `cleanlab` is powered by [provable guarantees](https://arxiv.org/abs/1911.00068) of exact noise estimation and label error finding in realistic cases when model output probabilities are erroneous. `cleanlab` supports multi-label, multiclass, sparse matrices, etc. By default, `cleanlab` requires no hyper-parameters.

`cleanlab` implements the family of theory and algorithms called [confident learning](https://arxiv.org/abs/1911.00068) with provable guarantees of exact noise estimation and label error finding (even when model output probabilities are noisy/imperfect).

`cleanlab` supports many classification tasks: multi-label, multiclass, sparse matrices, etc.

`cleanlab` is:

1.  **backed-by-theory** - Provable perfect label error finding in some realistic conditions.
2.  **fast** - Non-iterative, parallelized algorithms (e.g. < 1 second to find label issues in ImageNet with pre-computed probabilities)
3.  **general** - Works with any dataset, model, and framework, e.g., Tensorflow, PyTorch, sklearn, xgboost, etc.
--->

## Find label issues with PyTorch, Tensorflow, sklearn, xgboost, etc. in 1 line of code

```python
# Compute pred_probs (n x m matrix of predicted probabilities) on your own, with any classifier.
# Be sure you compute probs in a holdout/out-of-sample manner (e.g. via cross-validation)
# Here is an example that shows in detail how to compute pred_probs on CIFAR-10:
#    https://github.com/cleanlab/examples/tree/master/cifar10
# Now finding label issues is trivial with cleanlab... its one line of code.
# label issues are ordered by likelihood of being an error. First index is most likely error.
from cleanlab.filter import find_label_issues

ordered_label_issues = find_label_issues(
    labels=numpy_array_of_noisy_labels,
    pred_probs=numpy_array_of_predicted_probabilities,
    return_indices_ranked_by='normalized_margin', # Orders label issues
 )
```
<!--- 
**CAUTION:** Predicted probabilities from your model must be out-of-sample\! You should never provide predictions on the same datapoints used to train the model, as these will be overfit and unsuitable for finding label-errors. To obtain out-of-sample predicted probabilities for every datapoint in your dataset, you can use [cross-validation](https://machinelearningmastery.com/out-of-fold-predictions-in-machine-learning/). Alternatively, you can train your model on a separate dataset and you are only evaluating labels in data that was previously held-out.

Pre-computed **out-of-sample** predicted probabilities for CIFAR-10 train set are available: [here](https://github.com/cleanlab/examples/tree/master/cifar10#pre-computed-psx-for-every-noise--sparsity-condition).

## Learning with noisy labels in 3 lines of code

```python
from cleanlab.classification import LearningWithNoisyLabels
from sklearn.linear_model import LogisticRegression

# Wrap around any classifier. Yup, you can use sklearn/pyTorch/Tensorflow/FastText/etc.
lnl = LearningWithNoisyLabels(clf=LogisticRegression())
lnl.fit(X=X_train_data, labels=train_noisy_labels)
# Estimate the predictions you would have gotten by training with *no* label issues.
predicted_test_labels = lnl.predict(X_test)
```

Check out these [examples](https://github.com/cleanlab/examples) and [tests](https://github.com/cleanlab/cleanlab/tree/master/tests) (includes how to use other types of models).

## Learn cleanlab in 5min

New to `cleanlab`? Try out these easy tutorials:

1.  [Simple example of learning with noisy labels on the Iris dataset (multiclass classification)](https://github.com/cleanlab/examples/blob/master/iris_simple_example.ipynb).
2.  [Learning with noisy labels on CIFAR](https://github.com/cleanlab/examples/tree/master/cifar10)

## Use `cleanlab` with any model (Tensorflow, PyTorch, sklearn, xgboost, etc.)

All of the features of the `cleanlab` package work with **any model**. Yes, any model. Feel free to use PyTorch, Tensorflow, caffe2, scikit-learn, mxnet, etc. If you use a scikit-learn classifier, all `cleanlab` methods will work out-of-the-box. It’s also easy to use your favorite model from a non-scikit-learn package, just wrap your model into a Python class that inherits the `sklearn.base.BaseEstimator`:

``` python
from sklearn.base import BaseEstimator
class YourFavoriteModel(BaseEstimator): # Inherits sklearn base classifier
    def __init__(self, ):
        pass
    def fit(self, X, y, sample_weight=None):
        pass
    def predict(self, X):
        pass
    def predict_proba(self, X):
        pass
    def score(self, X, y, sample_weight=None):
        pass

# Now you can use your model with `cleanlab`. Here's one example:
from cleanlab.classification import LearningWithNoisyLabels
lnl = LearningWithNoisyLabels(clf=YourFavoriteModel())
lnl.fit(train_data, train_labels_with_errors)
```

#### Want to see a working example? [Here’s a compliant PyTorch MNIST CNN class](https://github.com/cleanlab/cleanlab/blob/master/cleanlab/models/mnist_pytorch.py)

As you can see [here](https://github.com/cleanlab/cleanlab/blob/master/cleanlab/models/mnist_pytorch.py), technically you don’t actually need to inherit from `sklearn.base.BaseEstimator`, as you can just create a class that defines `.fit()`, `.predict()`, and `.predict\_proba()`, but inheriting makes downstream scikit-learn applications like hyper-parameter optimization work seamlessly. See [cleanlab.classification.LearningWithNoisyLabels()](https://github.com/cleanlab/cleanlab/blob/master/cleanlab/classification.py#L106) for a fully compliant model.

Note, some libraries exists to do this for you. For PyTorch, check out the `skorch` Python library which will wrap your `pytorch` model into a `scikit-learn` compliant model.
--->

----

# Installation

Python 3.6+ are supported. Linux, macOS, and Windows are supported.

<!---
Stable release (pip):

``` bash
$ pip install cleanlab  # Using pip
```

Stable release (conda):

``` bash
$ conda install -c cleanlab cleanlab  # Using conda
```

Developer release:

``` bash
$ pip install git+https://github.com/cleanlab/cleanlab.git
```
--->
Install the following packages:

``` bash
$ conda install -c conda-forge snorkel sentence-transformers
$ conda install -c huggingface 
$ conda install -c huggingface tokenizers=0.10.1 transformers
$ conda install -c anaconda jupyter
```


## Reproducing Results in [Confident Learning paper](https://arxiv.org/abs/1911.00068)

### State of the Art Learning with Noisy Labels in CIFAR

A step-by-step guide to reproduce these results is available [here](https://github.com/cleanlab/examples/tree/master/cifar10). This guide is also a good tutorial for using cleanlab on any large dataset. You'll need to `git clone`
[confidentlearning-reproduce](https://github.com/cgnorthcutt/confidentlearning-reproduce) which contains the data and files needed to reproduce the CIFAR-10 results.

![](https://raw.githubusercontent.com/cleanlab/assets/master/cleanlab/cifar10_benchmarks.png)

Comparison of confident learning (CL), as implemented in `cleanlab`, versus seven recent methods for learning with noisy labels in CIFAR-10. Highlighted cells show CL robustness to sparsity. The five CL methods estimate label issues, remove them, then train on the cleaned data using [Co-Teaching](https://github.com/cleanlab/cleanlab/blob/master/cleanlab/coteaching.py).

Observe how cleanlab (i.e. the CL method) is robust to large sparsity in label noise whereas prior art tends to reduce in performance for increased sparsity, as shown by the red highlighted regions. This is important because real-world label noise is often sparse, e.g. a tiger is likely to be mislabeled as a lion, but not as most other classes like airplane, bathtub, and microwave.

### Find label issues in ImageNet

Use `cleanlab` to identify \~100,000 label errors in the 2012 ILSVRC ImageNet training dataset: [examples/imagenet](https://github.com/cleanlab/examples/tree/master/imagenet).

![](https://raw.githubusercontent.com/cleanlab/assets/master/cleanlab/imagenet_train_label_errors_32.jpg)

Label issues in ImageNet train set found via `cleanlab`. Label Errors are boxed in red. Ontological issues in green. Multi-label images in blue.

| `p(label︱y)` | y=0  | y=1  | y=2  | y=3  |
|--------------|------|------|------|------|
| label=0      | 0.55 | 0.01 | 0.07 | 0.06 |
| label=1      | 0.22 | 0.87 | 0.24 | 0.02 |
| label=2      | 0.12 | 0.04 | 0.64 | 0.38 |
| label=3      | 0.11 | 0.08 | 0.05 | 0.54 |

# ML Research with cleanlab

## `cleanlab` Core Package Components

1.  **cleanlab/classification.py** - [LearningWithNoisyLabels()](https://github.com/cleanlab/cleanlab/blob/master/cleanlab/classification.py#L106) class for learning with noisy labels.
2.  **cleanlab/count.py** - Estimates and fully characterizes all variants of label noise.
3.  **cleanlab/noise\_generation.py** - Generate mathematically valid synthetic noise matrices.
4.  **cleanlab/filter.py** - Finds the examples with label issues in a dataset.
5.  **cleanlab/rank.py** - Rank every example in a dataset with various label quality scores.

Many methods have default parameters not covered here. Check out the
method docstrings and our [full documentation](https://docs.cleanlab.ai/).

For additional details/notation, refer to [the Confident Learning paper](https://jair.org/index.php/jair/article/view/12125).

# Citation and Related Publications

`cleanlab` isn't just a github, it's based on peer-reviewed research. Here are the relevant papers to cite if you use this package:

The [confident learning paper](https://arxiv.org/abs/1911.00068):
 
    @article{northcutt2021confidentlearning,
        title={Confident Learning: Estimating Uncertainty in Dataset Labels},
        author={Curtis G. Northcutt and Lu Jiang and Isaac L. Chuang},
        journal={Journal of Artificial Intelligence Research (JAIR)},
        volume={70},
        pages={1373--1411},
        year={2021}
    }

## Other Resources

  - [Blogpost: Introduction to Confident Learning](https://l7.curtisnorthcutt.com/confident-learning)
  - [NeurIPS 2021 paper: Pervasive Label Errors in Test Sets Destabilize Machine Learning Benchmarks](https://arxiv.org/abs/2103.14749)

## License

Copyright (c) 2022 Carnegie Mellon University, Auton Lab.

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

See [MIT LICENSE](https://github.com/autonlab/KeyClass/blob/main/LICENSE) for details.


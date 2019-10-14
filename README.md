# Translate with Transformer

**Table of Contents**

[TOC]

This is a pytorch implementation of the transformer model adapted from the official repos of the [Annotated Transformer](https://github.com/harvardnlp/annotated-transformer).
To understand the model better, read the original paper:  [Vaswani et al., 2017](https://arxiv.org/abs/1706.03762), the [Annotated Transformer](https://nlp.seas.harvard.edu/2018/04/03/attention.html) or [see my blog](www.machinelearnit.com)

The code can be used for generating English-to-French and French-to-English translation models.
The code trains the translation models on the Europarl dataset (download it from [here:](https://www.statmt.org/europarl/) and place it to the ``data/`` directory.)

See the notebooks for each command for training and saving the models.

# Setup

## Installation

Create a new virtual environment and install packages.

```
conda create -n translate python
conda activate translate
```
If using cuda:
```
conda install pytorch cudatoolkit=10.0 -c pytorch
```

else:
```
conda install pytorch cpuonly -c pytorch
```
Next, install packages:

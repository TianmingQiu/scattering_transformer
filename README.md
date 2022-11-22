# SViT: Hybrid Vision Transformer Models With Scattering Transform
This work is published in [2022 IEEE 32nd International Workshop on Machine Learning for Signal Processing (MLSP)](https://ieeexplore.ieee.org/document/9943334)

## Table of Contents
0. [Release Notes](#Release-Notes)
0. [Introduction](#introduction)
0. [Prerequisites](#Prerequisites)
0. [License](#License)
0. [Bibtex](#Bibtex)

## Release Notes
- **Release 1.0**, (22.06.2022)
    - Git tag: MLSP-v1.0


## Introduction
![image](SViT_framework.png)
Overview of the model: we propose hybrid ViT models with scattering transform called Scattering Vision Transformer (SViT). More specifically, we investigate three tokenizations using scattering transform for ViT: patch-wise scattering tokens (SViTPatch), scattering image feature tokens (SViT-Image), and scattering frequency sub-band response tokens (SViT-Freq). 

## Prerequisites

#### Installation

*- Clone repository and install Python dependencies*
```sh
$ git clone https://github.com/TianmingQiu/scattering_transformer
$ cd scattering_transformer
$ pip install -r requirements.txt 
```

#### Initialization

*- Create local save folder and log folder*
```sh
$ cd scattering_transformer
$ mkdir checkpoint
$ mkdir log
```

*- Download the dataset*
```sh
$ cd input/dataset
```

#### Train Models

*- Configure the parameters of the model in the "custom_dataset.py" and "transforms.py" (if needed)*


*- Change the variable "DATA_TYPE" to the dataset you want to test into in the main function*






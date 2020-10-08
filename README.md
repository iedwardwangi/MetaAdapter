# Meta Language-Specific Layers in Multilingual Language Models

This repo contains the source codes for our paper

>[On Negative Interference in Multilingual Models: Findings and A Meta-Learning Treatment](http://arxiv.org/abs/2010.03017)

>Zirui Wang, Zachary C. Lipton, Yulia Tsvetkov

>EMNLP 2020

## Introduction

This repo contains code to train multilingual language models (XLM) that (1) contain language-specific layers, and (2) meta-learn these layers through gradient of gradient.

Language-specific layers are served as meta parameters, optimized using an iterative procedure. The goal is to remedy negative transfer in multilingual models through a meta training objective. Please see our paper for details.

## Dependencies

* Python 3
* [XLM](https://github.com/facebookresearch/XLM)
* [NumPy](http://www.numpy.org/)
* [PyTorch](http://pytorch.org/)

## Usage

The code is based on the official implementation of [XLM](https://github.com/facebookresearch/XLM). This repo only contains files that we modified from the original codebase.
To train a model, please merge code with the source code of XLM, and then follow the standard preprocessing and training instructions there.
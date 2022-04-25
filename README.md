# Rethinking-the-Usage-of-Normalization-and-Dropout-in-the-Training-of-Deep-Neural-Networks

Code Structure:

`models/` model definitions

`utils/` utils functions and dataset loaders

`train/` main training code

`res/` results, currently only to save trained models

`figures/` visualization results

## Image Domain Experiment

We are going to show results of 3 different kinds of models on 4 different datasets to prove the generalization property of IC layer. Currently, IC layer improves performance of VGG, GoogleNet and ResNet, but fails for DenseNet and MobileNet.

# Rethinking-the-Usage-of-Normalization-and-Dropout-in-the-Training-of-Deep-Neural-Networks

Code Structure:

`models/` model definitions

`utils/` utils functions and dataset loaders

`train/` main training code

`res/` results, currently only to save trained models

`figures/` visualization results

## Image Domain Experiment

We are going to show results of 3 different kinds of models on 4 different datasets to prove the generalization property of IC layer. Currently, IC layer improves performance of VGG, GoogleNet and ResNet, but fails for DenseNet and MobileNet.


## CLICK-THROUGH RATE PREDICTION TASK BASED ON WIDE & DEEP
Click-through rate (CTR) is a very important metric for evaluating the performance of online advertising or recommendation system. Some machine learning models will be used to predict CTR, whether the user will click or not. Wide Deep Model is the start-of-the-art model to predict CTR. To evaluate the performance of the IC layer on the CTR Task, we conducted extensive experiments based on Wide Deep Model.

Code: WideAndDeep_pytorch.ipynb
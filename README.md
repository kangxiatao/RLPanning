# Neural Network Panning

Neural Network Panning: Screening the Optimal Sparse Network Before Training

## Introduce

This project is a PyTorch implementation of the Panning paper, including several pruning before training methods tested in the paper.


## Structure
 - models: (lenet,resnet,vgg) Three models and the model class containing the mask
 - pruner: Pruning algorithm
 - runs: Output folder to store weights, evaluation information, etc.
 - utils: custom helper function


## Environment

The code has been tested to run on Python3.

Some package versions are as follows:
* torch == 1.7.1
* numpy == 1.18.5
* tensorboardX==2.4

## Run

* E.g. cifar10/vgg19 prune ratio: 90%
```
# Panning
python main.py --config 'cifar10/vgg19/90' --run 'test' --prune_mode 'panning'
```
```
# RLPanning
python main.py --config 'cifar10/vgg19/90' --run 'test' --prune_mode 'rlpanning'
```
```
# SynFlow
python main.py --config 'cifar10/vgg19/90' --run 'test' --rank_algo 'synflow' --prune_mode 'rank'
```

* E.g. mnist/lenet5 prune ratio: 99.9%
```
python main.py --config 'mnist/lenet5/99.9' --run 'test' --prune_mode 'panning'
```

- Model optional: ```lenet, vgg, resnet```

- Dataset optional: ```fashionmnist, mnist, cifar10, cifar100```(Other datasets need to be manually downloaded to the local)

- All parameters(The default parameters are determined by the configs.py):

    | Console Parameters | Remark |
    | :---- | :---- |
    | config = '' | # Select Dataset, Model, and Pruning Rate |
    | pretrained = '' | # Path to load pretrained model |
    | run = 'test' | # Experimental Notes |
    | rank_algo = 'snip' | # Choose a pruning algorithm (snip, grasp, synflow, snip_ori, grasp_ori)|
    | prune_mode = 'rank' | # Choose a pruning mode (panning, rlpanning, dense, rank, rank/random, rank/iterative)|
    | dp = '../Data' | # Modify the path of the dataset |
    | storage_mask = 0 | # Store the resulting mask |
    | save_model = 0 | # Store the Panning agent |


## ...
To be added ...


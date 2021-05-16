# Visual Sudoku Solver 

In this repo, a sudoku solver is designed to solve directly from images. Conditional GAN and Recurrent Relational Networks are used for this task.

## 1. cGAN

- [x] create a supervised dataset from limited labelled dataset

  - [x] K-Means - mini batch (2 variants)

    ~~InfoGANimage clustering methods, Siamese networks~~

- [x] architecture as per the original paper

- [x] evaluation metric - FID

- [x] use the original dataset, and retrain cGAN

- [x] generate lot of labelled dataset from the given (unlabelled) dataset for training a classifier network : LeNet

## 2. Recurrent Relational Network

- [x] RRN (as per paper with slight modification) ([reference](https://github.com/wDaniec/pytorch-RNN))

- [x] 73% digit-wise accuracy (and 0 board-wise accuracy) on dataset (from kmeans)

- [ ] Use policy gradients for additional 0-1 Loss 

## 3. Joint Training

1. [SatNet](https://arxiv.org/pdf/1905.12149.pdf) : idea of training classifier and rrn together (but this was possible as they had symbolic ground truth sudoku output tables)
2. Optimize loss function of RRN with two more loss functions on (same or tied weights) classifier as penalty terms 
3. we may use here cGAN with batchnorm or bit more "richer" architecture (wrt paper) .. ? (RRN is already modified)

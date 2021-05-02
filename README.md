# Visual Sudoku Solver 

In this repo, a sudoku solver is designed to solve directly from images. Conditional GAN and Recurrent Relational Networks are used for this task.

## 1. Dataset

create a supervised dataset from limited labelled dataset

- [x] K-Means
- [ ] image clustering methods
- [ ] Siamese networks
- [ ] put everything into python scripts

## 2. cGAN

- [x] architecture as per the original paper
- [x] evaluation metric - FID
- [ ] use the original dataset, and retrain cGAN
- [ ] generate lot of labelled dataset from the given (unlabelled) dataset for training a classifier network
- [ ] put everything into python scripts

## 3. Recurrent Relational Network

- [ ] scripts ready (implementation of RRN)
- [ ] retrain RRN on actual dataset (depends on Image clustering)
- [ ] Use data augmentation i.e. permutation as mentioned in paper

## 4. Joining all the dots together

- [ ] literature review 
- [ ] something about 0-1 loss (from RL paradigm)


# Visual Sudoku Solver 

In this repo, a sudoku solver is designed to solve directly from images. Conditional GAN and Recurrent Relational Networks are used for this task.

## 1. Dataset

create a supervised dataset from limited labelled dataset

- [ ] K-Means
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

- [x] scripts ready (implementation of RRN) ([rep referred](https://github.com/wDaniec/pytorch-RNN))
- [ ] retrain RRN (using 8x8 rules) on actual dataset (depends on Image clustering)
- [ ] Use data augmentation i.e. permutation as mentioned in paper
- [ ] Use policy gradients i.e. additional 0-1 loss

#### Experimental observations on 9x9 sudoku dataset

1. Using MLP with 3 layers instead of 4 gives faster training - rrn learns faster  
2. Using cell state instead of hidden state in finding message signals, is giving proper and/or fast learning curve
   1. if hidden state is used, then extremely slow learning!! 0 (completely correct) accuracy in 75 epochs vs 100% when cell state used
3. In repo 16 dimensional one hot encoding of cell content is used (the last 6 values are zero always) and this is fed into mlp for getting 96 dimensional x. Using a 16 dimensional embedding (learnable) introduced more parameters and the learning curve shows that it is slightly slower in learning. But this may be because of smaller dataset, i.e. we are using more richer model now, but on complete data the richer model might work well
   1. Using nn.Embedding.from_pretrained or nn.Linear showed no difference; we ll use former
4. Concatenating row_col embedding (one hot) slowed down learning but most likely because of small dataset
5. Even after using hidden state, but with only cell content embedding (not row and column), then all the three versions of RRN (repo code, copy_from_repo code, and our RRN code) work i.e. is able to learn (though slowly)
6. Also, detach() vs detach().clone() showed latter is better - possibly because clone() allows backprop; so sticking with latter for now

**note** : these observations are on 9x9 sudoku dataset and doesn't have learnt embeddings also. _Now we can use exact architecture given in the paper._



#### Experimental observations on 8x8 dataset

To try

1. less number of steps and more batch size

## 4. Joining all the dots together

- [ ] literature review : https://arxiv.org/pdf/1905.12149.pdf


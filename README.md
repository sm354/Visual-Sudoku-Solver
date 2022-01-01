# Visual Sudoku Solver 

We design a purely neural architecture to solve Sudoku, without using the rules of Sudoku in any manner. This work falls broadly in the space of **neuro-symbolic reasoning**. We use **cGAN**, **Unsupervised Data Augmentation**, and **Recurrent Relational Network (RRN)** to build this solver. We improve upon the baseline by **30%** accuracy using our inductive biases and training framework. The report can be found [here](./A2_report.pdf).


## **Problem Setting and Challenges:** 

Problem statement can be found [here](./A2_ps.pdf).

1. **8x8 Sudoku Board Images:**  

   ![0](https://user-images.githubusercontent.com/50492433/120084470-009d7280-c0ee-11eb-96a3-dbbf2ba3cd51.png) (unsolved) ![0 (1)](https://user-images.githubusercontent.com/50492433/120084473-02ffcc80-c0ee-11eb-9211-09192eb955ea.png) (solved)

   **Note** : these are 8x8 sudoku boards, where each column, each row and each block of size 2x4 is filled with 8 unique digits in the solved board, comprising of digits that are not neccessarily the actual digits (recognized by humans).

2. **Digit Classifier:** In order to give symbolic input to sudoku solver model, we extract all sub-images from the board and then use a classifier model to create 8x8=64 digits Sudoku board. 

​		**Available labelled data**:

![Screenshot 2021-05-30 at 2 29 05 AM](https://user-images.githubusercontent.com/50492433/120084551-d13b3580-c0ee-11eb-823b-69c899e5ae47.png)

3. **NOT using Sudoku Rules**: we solve the problem using very limited dataset, allow the neural network to understand the constraints of Sudoku, and solve the **Constrained Satisfaction Problem** completely from dataset. 

In the following sections, we break the problem in two parts: (1) creating a classifier, and (2) training and improving RRN to solve Sudoku.

## Classifier using c-GAN and Unsupervised Data Augmentation (UDA)

Since we are short on labelled (digit) dataset, we experiment with different clustering methods, but upon visual inspection we found that the given digits don't get clustered properly. Instead, we use unsupervised data augmentation to leverage the vast amount of unlabelled dataset (from sudoku board input-output images). Once the pseudo labels are obtained from UDA, we train a c-GAN model to remove the noise in labels. 

### Training LeNet using c-GAN & UDA

```bash
cd ./SemiSupervised_cGAN 
mkdir ./temp_saves #for saving the results

python uda.py --unlabelled_datapath <large-unlabelled-dataset-path> --supervised_datapath <small-supervised-dataset-path> --supervised_labels <path-of-labels-of-supervised-dataset> --output_labels <path-of-labelled-image-dataset-given-as-unlabelled-datapath> --output_classifier <path-of-output-classifier-using-UDA-method>

# use saved labels of uda classifier to train GAN
python train_cgan.py --root_path_to_save <directory-to-save-results> --traindatapath <large-unlabelled-dataset-path> --trainlabelspath  <path-of-labelled-image-dataset-given-as-unlabelled-datapath> --train_or_gen train --num_epochs 100

#generate 9k images in form of npy files and save as gen9k.npy and target9k.npy
python train_cgan.py --gen_model_pretr <trained-model-path-from prev step> --gen9k_path <path-to-generated-images> --target9k_path <path-to-generated-image-labels> --train_or_gen generate


#convert the generated npy images in png images,  and 9k real images to png images and save them and then calculate FID score
python numpy2images.py --savedir <directory-to-save-results> --numpy_images_file <path-to-generated-images> --num_images 9000

# calculate FID assuing we have gpu access ,for this step, you need to install the pytorch FID package
python -m pytorch_fid --device "cuda:0" <directory-to-save-results>/generated_images <path-to-directory-having-real-images>

# we also need a path to the directory having real images <path-to-directory-having-real-images> which can be used to get the FID score between real and generated images from our GAN

```

## Recurrent Relational Network (RRN)

Having the symbolic data obtained from the previous section, we can now train RRN model independently. There would be some level of noise in the symbolic data, and thus, it limits the ability of RRN to learn the constraints of Sudoku. However, we can also train both the classifier and RRN model in a joint framework, and we experimentally observed that this improves the performance of both of them. This stems from the fact that RRN learnt some rules of Sudoku and this information, when passed to classifier, helped it improve itself, and this goes on back and forth.

### Training RRN

```bash
run_solver.sh <path_to_train> <path_to_test_query> <path_to_sample_imgs> <path_to_out_csv>
```

### Training RRN + LeNet together

Similar to the earlier part but this time, the classifier that we get from UDA is fine tuned while training the RRN. The pretrained classifier and RRN are trained jointly so that both improve each other

```bash
run_solver.sh <path_to_train> <path_to_test_query> <path_to_sample_imgs> <path_to_out_csv> true
```

## Authors

- [Shubham Mittal](https://www.linkedin.com/in/shubham-mittal-6a8644165/)
- [Harman Singh](https://www.linkedin.com/in/harman-singh-4243ab180/)

Course assignment in Deep Learning course ([course webpage](https://www.cse.iitd.ac.in/~parags/teaching/2021/col870/)) taken by [Prof. Parag Singla](https://www.cse.iitd.ac.in/~parags/)
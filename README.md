# Visual Sudoku Solver 

We design a purely neural architecture to solve Sudoku, without using the rules of Sudoku in any manner. This work falls broadly in the space of **neuro-symbolic reasoning**. We use **cGAN**, **Unsupervised Data Augmentation**, and **Recurrent Relational Network (RRN)** to build this solver.

### **Problem Setting and Challenges:** 

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

### Quick Start

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

### Getting started without Joint Training

```bash
run_solver.sh <path_to_train> <path_to_test_query> <path_to_sample_imgs> <path_to_out_csv>
```

### With Joint training

Similar to the earlier part but this time, the classifier that we get from UDA is fine tuned while training the RRN. The pretrained classifier and RRN are trained jointly so that both improve each other

```bash
run_solver.sh <path_to_train> <path_to_test_query> <path_to_sample_imgs> <path_to_out_csv> true
```

- <path_to_train> directory has to sub directories, <path_to_train>/query/ and <path_to_train>/target/. Both these subdirectories have images of sudoku boards made of handwritten digits. Solution of the board <path_to_train>/query/n.png should be <path_to_train>/target/n.png where n is the number of the board (eg 0.png, 1.png ......)
- <path_to_test_query> has unsolved visual boards just like in <path_to_train>/query/ that will be solved after model is trained (for testing purposes)
- <path_to_sample_imgs> is a numpy file (.npy) of shape (10,784) having one labelled image of each class (digit)
- <path_to_out_csv> is where the result of solving the unsolved sudoku boards present in <path_to_test_query> will be stored in symbolic form (in the form of digits).

### Result

After joint training, we saw a boost in performance of RRN by 30% as compared to without joint training. Finally, we got a classifier with **82.24%** accuracy, and RRN model with **45.12%**. We have also incorporated **inductive biases** in the RRN model, details of which can be found in our [report](./A2_report.pdf).

## References

Martin Arjovsky, Soumith Chintala, and L ́eon Bottou. Wasserstein generative adversarial networks. In Proc. of ICML, 2017.

Ian Goodfellow, Jean Pouget-Abadie, Mehdi Mirza, Bing Xu, David Warde- Farley, Sherjil Ozair, Aaron Courville, and Yoshua Bengio. Generative adver- sarial nets. In Proc. of NeurIPS, 2014.

Ishaan Gulrajani, Faruk Ahmed, Martin Arjovsky, Vincent Dumoulin, and Aaron Courville. Improved training of wasserstein gans. In Proc. of NeuRIPS, 2017.

Martin Heusel, Hubert Ramsauer, Thomas Unterthiner, Bernhard Nessler, and Sepp Hochreiter. Gans trained by a two time-scale update rule converge to a local nash equilibrium. In Proc. of NeurIPS, 2017.

Tero Karras, Timo Aila, Samuli Laine, and Jaakko Lehtinen. Progressive grow- ing of gans for improved quality, stability, and variation. In Proc. of ICLR, 2018.

Y. Lecun. The mnist database of handwritten digits. http://yann.lecun.com/ exdb/mnist/, 2010.

Mehdi Mirza and Simon Osindero. Conditional generative adversarial nets. CoRR, abs/1411.1784, 2014.

Rasmus Berg Palm, Ulrich Paquet, and Ole Winther. Recurrent relational net- works. In Proc. of NeurIPS, 2018.

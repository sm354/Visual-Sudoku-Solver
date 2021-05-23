#!/bin/sh
### Set the job name (for your reference)
#PBS -N unetOgan
### Set the project name, your department code by default
#PBS -P ee
### Request email when job begins and ends
#PBS -m bea
### Specify email address to use for notification.
#PBS -M $USER@iitd.ac.in
####
#PBS -l select=1:ngpus=1

### Specify "wallclock time" required for this job, hhh:mm:ss
#PBS -l walltime=10:00:00

#PBS -l software=python
# After job starts, must goto working directory. 
# $PBS_O_WORKDIR is the directory from where the job is fired. 
echo "==============================="
echo $PBS_JOBID
cat $PBS_NODEFILE
echo "==============================="
cd $PBS_O_WORKDIR

module purge
module load apps/anaconda/3
# module load apps/pytorch/1.6.0/gpu/anaconda3

#minibatch-kmeans
# python clustering/clustering.py --savedir results/kmeans-minibatch --query_datapath ../sudoku_array_data/query_64k_images.npy --target_datapath ../sudoku_array_data/target_64k_images.npy --oneshot_datapath "../Assignment 2/sample_images.npy" --nclusters 8 --output_label_file results/kmeans-minibatch/kmeans_mb_t8c_labels.npy --output_oneshot_label_file results/kmeans-minibatch/kmeans_mb_t8c_oneshot_labels.npy --method minbatch-kmeans
# python clustering/clustering.py --savedir results/kmeans-minibatch --query_datapath ../sudoku_array_data/query_64k_images.npy --target_datapath ../sudoku_array_data/target_64k_images.npy --oneshot_datapath "../Assignment 2/sample_images.npy" --nclusters 9 --output_label_file results/kmeans-minibatch/kmeans_mb_qt9c_labels.npy --output_oneshot_label_file results/kmeans-minibatch/kmeans_mb_qt9c_oneshot_labels.npy --method minbatch-kmeans

#pytorch-kmeans
# /usr/bin/time --verbose python clustering/clustering.py --savedir results/kmeans-pytorch --query_datapath ../sudoku_array_data/query_64k_images.npy --target_datapath ../sudoku_array_data/target_64k_images.npy --oneshot_datapath "../Assignment 2/sample_images.npy" --nclusters 8 --output_label_file results/kmeans-pytorch/kmeans_pyt_t8c_labels.npy --output_oneshot_label_file results/kmeans-pytorch/kmeans_pyt_t8c_oneshot_labels.npy --method pytorch-kmeans
# python clustering/clustering.py --savedir results/kmeans-pytorch --query_datapath ../sudoku_array_data/query_64k_images.npy --target_datapath ../sudoku_array_data/target_64k_images.npy --oneshot_datapath "../Assignment 2/sample_images.npy" --nclusters 9 --output_label_file results/kmeans-pytorch/kmeans_pyt_qt9c_labels.npy --output_oneshot_label_file results/kmeans-pytorch/kmeans_pyt_qt9c_oneshot_labels.npy --method pytorch-kmeans

#minibatch-kmeans-sampled
# python clustering/clustering.py --savedir results/kmeans-sampled_15k --query_datapath ../sudoku_array_data/query_64k_images.npy --target_datapath ../sudoku_array_data/target_64k_images.npy --oneshot_datapath "../Assignment 2/sample_images.npy" --nclusters 9 --output_label_file results/kmeans-sampled_15k/kmeans_sampled_qt9c_labels.npy --output_oneshot_label_file results/kmeans-sampled_15k/kmeans_sampled_qt9c_oneshot_labels.npy --method minibatch-kmeans-sampled --sampled_X_path results/kmeans-sampled_15k/dataX_kmeans_sampled_qt9c.npy
# python clustering/clustering.py --savedir results/kmeans-sampled_15k_10each --query_datapath ../sudoku_array_data/query_64k_images.npy --target_datapath ../sudoku_array_data/target_64k_images.npy --oneshot_datapath "../Assignment 2/sample_images.npy" --nclusters 9 --output_label_file results/kmeans-sampled_15k_10each/kmeans_sampled_qt9c_labels.npy --output_oneshot_label_file results/kmeans-sampled_15k_10each/kmeans_sampled_qt9c_oneshot_labels.npy --method minibatch-kmeans-sampled --sampled_X_path results/kmeans-sampled_15k_10each/dataX_kmeans_sampled_qt9c.npy

#minibatch-kmeans on vgg data
# python clustering/clustering.py --savedir results/kmeans-vgg --query_datapath results/vggnet_embeddings/X_query_target_vggnet.npy --oneshot_datapath results/vggnet_embeddings/oneshot_data_vggnet.npy --nclusters 9 --output_label_file results/kmeans-vgg/kmeans_vgg_qt9c_labels.npy --output_oneshot_label_file results/kmeans-vgg/kmeans_vgg_qt9c_oneshot_labels.npy --method minibatch-kmeans


#make data
# python load_sudoku_data.py --train_datapath "../Assignment 2/visual_sudoku/train" --target_array_file "../sudoku_array_data/target_64k_images.npy" --query_array_file "../sudoku_array_data/query_64k_images.npy" --query_target_array_file "../sudoku_array_data/query_target_64k_images.npy"

#saving labels as symbolic sudoku
# python label2symbolic_sudoku.py --labels_path results/kmeans-minibatch/kmeans_mb_qt9c_labels.npy --output_symbolic_sud_path results/symbolic_data/symbolic_sudoku_kmeans_mb_tq.npy

#train GAN
#--on query data
# python cGAN/train_cgan.py --root_path_to_save results/GAN_out/E9_query_150epochs --traindatapath ../sudoku_array_data/query_64k_images.npy --trainlabelspath results/kmeans-minibatch/kmeans_mb_qt9c_labels.npy --train_or_gen train --num_epochs 150
#--onsampledkmeans data
# python cGAN/train_cgan.py --root_path_to_save results/GAN_out/E13_nodrop_sampledkmeans_200epochs --traindatapath results/kmeans-sampled_15k/dataX_kmeans_sampled_qt9c.npy --trainlabelspath results/kmeans-sampled_15k/kmeans_sampled_qt9c_labels.npy --train_or_gen train --num_epochs 200
#--on query+target data
# python cGAN/train_cgan.py --root_path_to_save results/GAN_out/E14_nodrop_querandtarget_150epochs --traindatapath ../sudoku_array_data/query_target_64k_images.npy --trainlabelspath results/kmeans-minibatch/kmeans_mb_qt9c_labels.npy --train_or_gen train --num_epochs 150
#------------SGD on query +target
# python cGAN/train_cgan.py --root_path_to_save results/GAN_out/E15_sgd0.0001_drop_querandtarget_150epochs --traindatapath ../sudoku_array_data/query_target_64k_images.npy --trainlabelspath results/kmeans-minibatch/kmeans_mb_qt9c_labels.npy --train_or_gen train --num_epochs 150
# python cGAN/train_cgan.py --root_path_to_save results/GAN_out/E16_sgdgen5e3dis1e3_drop_querandtarget_150epochs --traindatapath ../sudoku_array_data/query_target_64k_images.npy --trainlabelspath results/kmeans-minibatch/kmeans_mb_qt9c_labels.npy --train_or_gen train --num_epochs 150
# python cGAN/train_cgan.py --root_path_to_save results/GAN_out/E17_sgdgen1e2dis1e2_drop_querandtarget_150epochs --traindatapath ../sudoku_array_data/query_target_64k_images.npy --trainlabelspath results/kmeans-minibatch/kmeans_mb_qt9c_labels.npy --train_or_gen train --num_epochs 150


#validate/generate from gan generator in form of npy files
# python cGAN/train_cgan.py --gen_model_pretr results/GAN_out/E8_querandtarget_150epochs/gen_trained.pth --gen9k_path results/GAN_out/E8_querandtarget_150epochs/gen9k.npy --target9k_path results/GAN_out/E8_querandtarget_150epochs/target9k.npy --train_or_gen generate

#convert numpy generated images to png images 
# python numpy2images.py --savedir results/GAN_out/E8_querandtarget_150epochs/gen_imgs --numpy_images_file results/GAN_out/E8_querandtarget_150epochs/gen9k.npy  --num_images 9000
#convert real images-numpy to jpg
# python numpy2images.py --savedir results/real_images_9k --numpy_images_file ../sudoku_array_data/query_target_64k_images.npy --num_images 9000


#train classifier for arabic mnist
#---for kmeans
# python classifier_arabicmnist.py --root_path_to_save results/classifier/E7_kmeans_train_query_9class --traindatapath ../sudoku_array_data/query_64k_images.npy --trainlabelspath results/kmeans-minibatch/kmeans_mb_qt9c_labels.npy --num_classes 9 --targetdatapath ../sudoku_array_data/target_64k_images.npy
# python Joint_training/classifier_arabicmnist.py --root_path_to_save results/classifier/testing_classifier --traindatapath ../sudoku_array_data/query_target_64k_images.npy --trainlabelspath results/kmeans-minibatch/kmeans_mb_qt9c_labels.npy --num_classes 9 --targetdatapath ../sudoku_array_data/target_64k_images.npy

#--for sampled kmeans
# python classifier_arabicmnist.py --root_path_to_save results/classifier/E10_sampledkmeans_train_query_9class --traindatapath results/kmeans-sampled_15k/dataX_kmeans_sampled_qt9c.npy --trainlabelspath results/kmeans-sampled_15k/kmeans_sampled_qt9c_labels.npy --num_classes 9 --targetdatapath ../sudoku_array_data/target_64k_images.npy

#vggnet embeddigns
# python vggnet_embeddings.py


#fid 
# python -m pytorch_fid --device "cuda:0" results/GAN_out/E8_querandtarget_150epochs/gen_imgs_28/ results/real_images_9k_28/

#NOTE
# The job line is an example : users need to change it to suit their applications
# The PBS select statement picks n nodes each having m free processors
# OpenMPI needs more options such as $PBS_NODEFILE 

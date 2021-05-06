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
#PBS -l walltime=18:00:00

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

#minibatch-kmeans
# python clustering/clustering.py --savedir results/kmeans-minibatch --query_datapath ../sudoku_array_data/query_64k_images.npy --target_datapath ../sudoku_array_data/target_64k_images.npy --oneshot_datapath "../Assignment 2/sample_images.npy" --nclusters 8 --output_label_file results/kmeans-minibatch/kmeans_mb_t8c_labels.npy --output_oneshot_label_file results/kmeans-minibatch/kmeans_mb_t8c_oneshot_labels.npy --method minbatch-kmeans
# python clustering/clustering.py --savedir results/kmeans-minibatch --query_datapath ../sudoku_array_data/query_64k_images.npy --target_datapath ../sudoku_array_data/target_64k_images.npy --oneshot_datapath "../Assignment 2/sample_images.npy" --nclusters 9 --output_label_file results/kmeans-minibatch/kmeans_mb_qt9c_labels.npy --output_oneshot_label_file results/kmeans-minibatch/kmeans_mb_qt9c_oneshot_labels.npy --method minbatch-kmeans

#pytorch-kmeans
# /usr/bin/time --verbose python clustering/clustering.py --savedir results/kmeans-pytorch --query_datapath ../sudoku_array_data/query_64k_images.npy --target_datapath ../sudoku_array_data/target_64k_images.npy --oneshot_datapath "../Assignment 2/sample_images.npy" --nclusters 8 --output_label_file results/kmeans-pytorch/kmeans_pyt_t8c_labels.npy --output_oneshot_label_file results/kmeans-pytorch/kmeans_pyt_t8c_oneshot_labels.npy --method pytorch-kmeans
# python clustering/clustering.py --savedir results/kmeans-pytorch --query_datapath ../sudoku_array_data/query_64k_images.npy --target_datapath ../sudoku_array_data/target_64k_images.npy --oneshot_datapath "../Assignment 2/sample_images.npy" --nclusters 9 --output_label_file results/kmeans-pytorch/kmeans_pyt_qt9c_labels.npy --output_oneshot_label_file results/kmeans-pytorch/kmeans_pyt_qt9c_oneshot_labels.npy --method pytorch-kmeans

#make data
# python load_sudoku_data.py --train_datapath "../Assignment 2/visual_sudoku/train" --target_array_file "../sudoku_array_data/target_64k_images.npy" --query_array_file "../sudoku_array_data/query_64k_images.npy"

#saving labels as symbolic sudoku
# python label2symbolic_sudoku.py --labels_path results/kmeans-minibatch/kmeans_mb_qt9c_labels.npy --output_symbolic_sud_path results/symbolic_data/symbolic_sudoku_kmeans_mb_tq.npy

#train GAN
python cGAN/train_cgan.py --root_path_to_save results/GAN_out/E2_150epochs --traindatapath ../sudoku_array_data/query_64k_images.npy --trainlabelspath results/kmeans-minibatch/kmeans_mb_qt9c_labels.npy --train_or_gen train --num_epochs 150

#validate/generate from gan generator

#NOTE
# The job line is an example : users need to change it to suit their applications
# The PBS select statement picks n nodes each having m free processors
# OpenMPI needs more options such as $PBS_NODEFILE 

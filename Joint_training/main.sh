#!/bin/sh
### Set the job name (for your reference)
#PBS -N four
### Set the project name, your department code by default
#PBS -P ee
### Request email when job begins and ends
#PBS -m bea
### Specify email address to use for notification.
#PBS -M $USER@iitd.ac.in
####
#PBS -l select=1:ngpus=1

### Specify "wallclock time" required for this job, hhh:mm:ss
#PBS -l walltime=08:00:00

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


# python main.py --data_dir ../symbolic_sudoku_kmeans_mb_tq.npy --num_epochs 25 --num_steps 20 --exp_name rrn_20steps --savemodel ./saved_models/ --saveplot ./saved_results/

#joint training
python joint_train_algo2.py --lr_classifier 5e-4 --lr_rrn 2e-3 --data_dir ../../sudoku_array_data/query_target_64k_images.npy --pretr_classifier ../results/classifier/E9_kmeans_train_query_9class/classifier_trained.pth --loss_reg yes --num_epochs 100 --num_steps 20 --exp_name E_ --savemodel ./saved_models/ --saveplot ./saved_results/
python joint_train_algo3.py --lr_classifier 5e-4 --lr_rrn 2e-3 --lreg_factor 10 --oneshot_file "../../Assignment 2/sample_images.npy" --data_dir ../../sudoku_array_data/query_target_64k_images.npy --pretr_classifier ../results/classifier/E9_kmeans_train_query_9class/classifier_trained.pth --loss_reg yes --num_epochs 100 --num_steps 20 --exp_name E_ --savemodel ./saved_models/ --saveplot ./saved_results/


#NOTE
# The job line is an example : users need to change it to suit their applications
# The PBS select statement picks n nodes each having m free processors
# OpenMPI needs more options such as $PBS_NODEFILE
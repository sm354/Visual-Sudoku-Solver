# commandline arguments
# datapaths for query and train data (numpy arrays for now, images later)
# method name

import argparse

import numpy as np
import os
import matplotlib.pyplot as plt
import PIL.Image as Image
import torch
from sklearn.cluster import MiniBatchKMeans, KMeans
from sklearn import decomposition
import torch


def load_data(query_data, target_data, oneshot_data_path, nclusters):

	print("loading data--------------------")
	if(nclusters == 8):
		X = np.load(target_data)
	elif(nclusters == 9):
		X_query=np.load(query_data)
		X_target=np.load(target_data)
		X = np.concatenate((X_query, X_target))
	
	oneshot_data = np.load(oneshot_data_path)
	X = (X)/255. #normalization
	X = X.reshape((-1,28*28)) #shape (640k or 1280k, 784)
	oneshot_data = oneshot_data.reshape((-1, 28*28))/(255.) #shape 10, 784
	# x_oneshot_target = x_oneshot[:-1] #from 0th class to 8th class, 9th dropped as its no where in the images i THINK

	print('shape of X = ', X.shape)
	print('shape of x_oneshot = ', oneshot_data.shape)

	print('X = \n', X)
	print('x_oneshot = \n', oneshot_data)

	return X, oneshot_data


def perform_minibatchkmeans(X, x_oneshot_data, nclusters):
	# custering using minibatch kmeans
	if(nclusters == 8):
		oneshot_data = x_oneshot_data[1:-1]
	elif(nclusters == 9):
		oneshot_data = x_oneshot_data[:-1]
	#note nclusters should be same as shape[0] of x_oneshot_data
	kmeans = MiniBatchKMeans(n_clusters=nclusters, init = oneshot_data, random_state=0, batch_size = 1000).fit(X)
	
	cluster_centers = kmeans.cluster_centers_
	labels  = kmeans.labels_

	print('cluster-centers = \n', cluster_centers)
	print('cluster-centers shape= ', cluster_centers.shape)
	print('label shape = ', labels.shape)

	# predicting labels of one shot data, note we started of initisalizing cluster centers using this data only
	predicted_clusters_oneshot_data = kmeans.predict(oneshot_data)

	if(nclusters == 8): #add one to make it correct class number if we are clustering only 8 classes, no need to do anything if 9 classes
		labels, predicted_clusters_oneshot_data = labels+1, predicted_clusters_oneshot_data+1

	print("labels of one shot data = ", predicted_clusters_oneshot_data)

	return labels, cluster_centers, predicted_clusters_oneshot_data


def perform_pytorchkmeans(X, x_oneshot_data, nclusters):
	from kmeans_pytorch import kmeans as kmeans_pt
	from kmeans_pytorch import kmeans_predict as kmeans_predict_pt
	device='cuda:0' if torch.cuda.is_available() else 'cpu'

	if(nclusters == 8):
		oneshot_data = x_oneshot_data[1:-1]
		print("h")
	elif(nclusters == 9):
		print("here")
		oneshot_data = x_oneshot_data[:-1]

	X = torch.tensor(X)
	oneshot_data = torch.tensor(oneshot_data)

	labels, cluster_centers = kmeans_pt(X=X, num_clusters=nclusters, distance='euclidean', device=device)

	print('cluster-centers = \n', cluster_centers)
	print('cluster-centers shape= ', cluster_centers.shape)
	print('label shape = ', labels.shape)

	# predicting labels of one shot data, note we started of initisalizing cluster centers using this data only
	predicted_clusters_oneshot_targets = kmeans_predict_pt(oneshot_data, cluster_centers, 'euclidean', device=device)

	# convert to numpy arrays
	labels, cluster_centers, predicted_clusters_oneshot_targets = labels.numpy(), cluster_centers.numpy(), predicted_clusters_oneshot_targets.numpy()

	if(nclusters == 8): #add one to make it correct class number if we are clustering only 8 classes, no need to do anything if 9 classes
		labels, predicted_clusters_oneshot_targets = labels+1, predicted_clusters_oneshot_targets+1

	print("labels of one shot data = ", predicted_clusters_oneshot_targets)

	# # convert class numbers according to oneshot_data
	# oneshot_labels2 = {}
	# for i in range(predicted_clusters_oneshot_data.shape[0]):
	# 	label2_oneshot_labels[]

	return labels, cluster_centers, predicted_clusters_oneshot_targets

def save_cluster_labels(labels, oneshot_labels, labels_output_path, oneshot_labels_output_path):
	np.save(labels_output_path, labels)
	np.save(oneshot_labels_output_path, oneshot_labels)

if __name__ == "__main__":

	parser = argparse.ArgumentParser()
	parser.add_argument('--savedir', type=str, default = None)

	parser.add_argument('--query_datapath', type=str, default = None)
	parser.add_argument('--target_datapath', type=str, default = None)
	parser.add_argument('--oneshot_datapath', type=str, default = None)
	parser.add_argument('--nclusters', type=int, default = None)
	parser.add_argument('--output_label_file', type=str, default = None)
	parser.add_argument('--output_oneshot_label_file', type=str, default = None)
	parser.add_argument('--method', type=str)
	args=parser.parse_args()

	if not os.path.exists(args.savedir):
		os.makedirs(args.savedir)

	#loading data
	X, x_oneshot_data = load_data(args.query_datapath, args.target_datapath, args.oneshot_datapath, args.nclusters)

	#performing clustering
	if(args.method == 'minbatch-kmeans'):
		labels, cluster_centers, oneshot_labels = perform_minibatchkmeans(X, x_oneshot_data, args.nclusters)

	elif(args.method == 'pytorch-kmeans'):
		labels, cluster_centers, oneshot_labels  = perform_pytorchkmeans(X, x_oneshot_data, args.nclusters)

	# elif(args.method == 'n2d'):
	# 	labels, cluster_centers = perform_n2d_clustering(X, x_oneshot_data, args.nclusters)

	save_cluster_labels(labels, oneshot_labels, args.output_label_file, args.output_oneshot_label_file)
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


def load_data(query_data, target_data, oneshot_data_path):

	print("loading data--------------------")
	if(query_data is None and target_data is not None):
		X = np.load(target_data)
	elif(query_data is not None and target_data is None):
		X = np.load(query_data)
	else:
		X_target=np.load(query_data)
		X_query=np.load(target_data)
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

	#note nclusters should be same as shape[0] of x_oneshot_data
	kmeans = MiniBatchKMeans(n_clusters=nclusters, init = x_oneshot_data, random_state=0, batch_size = 200).fit(X)
	
	cluster_centers = kmeans.cluster_centers_
	labels  = kmeans.labels_

	print('cluster-centers = \n', cluster_centers)
	print('cluster-centers shape= ', cluster_centers.shape)
	print('label shape = ', labels.shape)

	# predicting labels of one shot data, note we started of initisalizing cluster centers using this data only
	predicted_clusters_oneshot_data = kmeans.predict(x_oneshot_data)
	print("labels of one shot data = ", predicted_clusters_oneshot_data)

	return labels, cluster_centers


def perform_pytorchkmeans(X, x_oneshot_data, nclusters, device):
	from kmeans_pytorch import kmeans_pt, kmeans_predict_pt
	device='cuda:0' if torch.cuda.is_available() else 'cpu'

	labels, cluster_centers = kmeans_pt(X=X, num_clusters=nclusters, distance='euclidean', device=device)

	print('cluster-centers = \n', cluster_centers)
	print('cluster-centers shape= ', cluster_centers.shape)
	print('label shape = ', labels.shape)

	# predicting labels of one shot data, note we started of initisalizing cluster centers using this data only
	predicted_clusters_oneshot_targets = kmeans_predict_pt(x_oneshot_data, cluster_centers, 'euclidean', device=device)
	
	print("labels of one shot data = ", predicted_clusters_oneshot_targets)

	return labels, cluster_centers

def save_cluster_labels(labels, labels_output_path):
	np.save(labels_output_path, labels)

if __name__ == "__main__":

	parser = argparse.ArgumentParser()
	parser.add_argument('--query_datapath', type=str, default = None)
	parser.add_argument('--target_datapath', type=str, default = None)
	parser.add_argument('--oneshot_datapath', type=str, default = None)
	parser.add_argument('--nclusters', type=int, default = None)
	parser.add_argument('--output_label_file', type=str, default = None)
	parser.add_argument('--method', type=str)
	args=parser.parse_args()


	#loading data
	X, x_oneshot_data = load_data(args.query_datapath, args.target_datapath, args.oneshot_datapath)

	#performing clustering
	if(args.method == 'minbatch-kmeans'):
		labels, cluster_centers = perform_minibatchkmeans(X, x_oneshot_data, args.nclusters)

	elif(args.method == 'pytorch-kmeans'):
		labels, cluster_centers  = perform_pytorchkmeans(X, x_oneshot_data, args.nclusters)

	# elif(args.method == 'n2d'):
	# 	labels, cluster_centers = perform_n2d_clustering(X, x_oneshot_data, args.nclusters)

	save_cluster_labels(labels, args.output_label_file)
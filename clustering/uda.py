import argparse

import numpy as np
import os
import matplotlib.pyplot as plt
import PIL.Image as Image
import torch
from sklearn.cluster import MiniBatchKMeans, KMeans
from sklearn import decomposition
import torch
from sklearn.metrics import pairwise_distances


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
	parser.add_argument('--sampled_X_path', type=str, default = None)
	args=parser.parse_args()

	if not os.path.exists(args.savedir):
		os.makedirs(args.savedir)

	#loading data
	X, x_oneshot_data = load_data(args.query_datapath, args.target_datapath, args.oneshot_datapath, args.nclusters)

	#performing clustering
	if(args.method == 'minbatch-kmeans'):
		labels, cluster_centers, oneshot_labels = perform_minibatchkmeans(X, x_oneshot_data, args.nclusters)

	if(args.method == 'minibatch-kmeans-sampled'):
		labels, cluster_centers, oneshot_labels, data_X = perform_minibatchkmeans_sampled(X, x_oneshot_data, args.nclusters)
		np.save(args.sampled_X_path, data_X)

	elif(args.method == 'pytorch-kmeans'):
		labels, cluster_centers, oneshot_labels  = perform_pytorchkmeans(X, x_oneshot_data, args.nclusters)
		
	# elif(args.method == 'n2d'):
	# 	labels, cluster_centers = perform_n2d_clustering(X, x_oneshot_data, args.nclusters)

	save_cluster_labels(labels, oneshot_labels, args.output_label_file, args.output_oneshot_label_file)
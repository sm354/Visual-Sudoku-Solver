import numpy as np

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--labels_path', type=str, default = None)
parser.add_argument('--output_symbolic_sud_path', type=str, default = None)

args=parser.parse_args()

labels = np.load(args.labels_path)
labels = labels.reshape((labels.shape[0]//64, -1))
labels = labels.reshape((-1, 8, 8))
print("label shape = {}".format(labels.shape))
#labels is of size 20000, 8, 8 now
np.save(args.output_symbolic_sud_path, labels) #tq for target query, mb for minibatch
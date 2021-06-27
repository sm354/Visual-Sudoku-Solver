"""for calculating the goodness of clustering"""
import numpy as np


def is_set_correct(array):
    # print(array)
    # print(set(array))
    if len(set(array)) >= 8:
        return True
    return False


def clustering_accuracy(labels):
    #labels are of shape (totalsmall images in all sudoku which is divisible by 64,)
    labels = labels.reshape((labels.shape[0]//64, -1))
    labels = labels.reshape((-1, 8, 8))
    print(labels.shape)
    print(labels[0])
    # print(labels[10000])

    subatomic_correct = 0

    correct = 0
    total = 0
    #now we have labels of correct shape
    final_bool_arr = np.array([True for i in range(labels.shape[0])])
    for i in range(8):
        j, k = (i // 4) * 4, (i % 2) * 2
        # if(np.all(np.apply_along_axis(is_set_correct, axis = 1, arr = labels[:, :, i])) == True or np.all(np.apply_along_axis(is_set_correct, axis = 1, arr = labels[:, i, :])) == True or np.all(np.apply_along_axis(is_set_correct, axis = 1, arr = labels[:, k:k+2, j:j+4].reshape(-1, 8))) !=True ):
        #   correct+=1
        # total+=1

        arr1 = np.apply_along_axis(is_set_correct, axis = 1, arr = labels[:, :, i])
        arr2 = np.apply_along_axis(is_set_correct, axis = 1, arr = labels[:, i, :])
        arr3 = np.apply_along_axis(is_set_correct, axis = 1, arr = labels[:, k:k+2, j:j+4].reshape(-1, 8))
        arr = arr1*arr2*arr3
        assert(arr.shape[0] == labels.shape[0] and len(arr.shape) == 1)
        final_bool_arr *= arr
        subatomic_correct += arr1.sum() + arr2.sum() + arr3.sum()
        # print(subatomic_correct)

        # if len(set(labels[i,:])) != 9 or len(set(grid[:,i])) != 9 or len(set(grid[j:j+3, k:k+3].ravel())) != 9:
        #   return False
    return final_bool_arr.sum()/final_bool_arr.shape[0], subatomic_correct/(3*8*labels.shape[0])


labels = np.load("/home/ee/btech/ee1180957/scratch/Harman/DL-ASS2/COL870-Assignment-2/results/kmeans-minibatch/kmeans_mb_qt9c_labels.npy")
# labels = np.load("/home/ee/btech/ee1180957/scratch/Harman/DL-ASS2/COL870-Assignment-2/results/kmeans-minibatch/kmeans_mb_t8c_labels.npy")
# labels = np.load("results/kmeans-minibatch/kmeans_mb_t8c_labels.npy")

# labels = np.load("results/kmeans-pytorch/kmeans_pyt_t8c_labels.npy")

# labels = np.load("/home/ee/btech/ee1180957/scratch/Harman/DL-ASS2/COL870-Assignment-2/results/kmeans-sampled_15k/kmeans_sampled_qt9c_labels.npy")


print(labels[0:1000])
print(labels.shape)
# print(clustering_accuracy(labels[:64000]))
print(clustering_accuracy(labels[640000:]))
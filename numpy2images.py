import numpy as np
import matplotlib.pyplot as plt
import os
import PIL.Image as Image
import torch.optim as optim
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--savedir', type=str, default = None)
    parser.add_argument('--numpy_images_file', type=str, default = None)
    parser.add_argument('--num_images', type=int, default = None)
    args=parser.parse_args()
    
    if not os.path.exists(args.savedir):
        os.makedirs(args.savedir)

    #convert numpy file to images and select num_images from them and save in savedir
    np_img = np.load(args.numpy_images_file)

    #shuffle images and take the first num_images
    shuffler = np.random.permutation(np_img.shape[0])
    np_img = np_img[shuffler]
    np_img = np_img[:args.num_images]
    np_img_tosave = np_img.reshape(-1, 28, 28)

    for img_index in range(args.num_images):
        # print(np_img_tosave.shape)
        y = np_img_tosave[img_index]
        y = y.reshape((28,28))
        plt.imshow(y, cmap='gray')
        plt.axis('off')
        plt.savefig(os.path.join(args.savedir, "{}.png".format(img_index)))
        plt.close()
        # plt.imsave(os.path.join(args.savedir, "{}.png".format(img_index)), y)

        # im = Image.fromarray(np_img_tosave[img_index]).convert("L")
        # im.save(os.path.join(args.savedir, "{}.png".format(img_index)))
        # if(img_index<5):
        #     break

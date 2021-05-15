# from sklearn.datasets import fetch_openml
import numpy as np
import matplotlib.pyplot as plt
import os
import PIL.Image as Image
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, TensorDataset
from IPython import display
import torch.optim as optim
import argparse


def sample_imgs(generator, device, numclasses):
    # generator.eval()
    with torch.no_grad():
        z=torch.rand(numclasses,100).to(device)
        z_labels=torch.tensor([i for i in range(numclasses)]).to(device)
        x_fake=generator(z,z_labels)
    x_fake=x_fake.cpu().numpy()
    z_labels=z_labels.cpu().numpy()

    plt.figure(figsize=(15.,30.))
    for ind in range(numclasses):
        plt.subplot(1,numclasses,ind+1)
        plt.title(z_labels[ind])
        plt.imshow(x_fake[ind].reshape((28,28)),cmap='gray')
        plt.axis('off')
    plt.show()
    # generator.train()

def sample_img_grid(generator, device, numclasses, img_savepath = None):
    # generator.eval()
    cols = numclasses
    rows = numclasses
    fig1, f1_axes = plt.subplots(ncols=cols, nrows=rows, figsize=(10,10))
    for index in range(rows):

        for ind in range(cols):
            with torch.no_grad():
                z=torch.rand(1,100).to(device)
                # print(z)
                z_labels=torch.tensor([index]).to(device)
                x_fake=generator(z,z_labels)
            x_fake=x_fake.cpu().numpy()
            z_labels=z_labels.cpu().numpy()

            f1_axes[index, ind].imshow(x_fake[0].reshape((28,28)),cmap='gray')
            f1_axes[index, ind].set_axis_off()
    # plt.show()

    if(img_savepath is not None):
        plt.savefig(os.path.join(img_savepath, "gen_image_grid.jpg"), dpi = 100)
    plt.close()

    #save the figure
    # generator.train()

class Maxout_layer(nn.Module):
    def __init__(self, input_dim=784, output_dim=240, pieces=5):
        super(Maxout_layer, self).__init__()
        self.params = torch.nn.ParameterList()
        self.fc_layers=[nn.Linear(in_features=input_dim,out_features=output_dim) for i in range(pieces)]
        for layer in self.fc_layers:
            self.params.extend(list(layer.parameters()))

    def forward(self,inp): # inp.shape=batch_size,784
        op=[fc_layer.to(inp.device)(inp) for fc_layer in self.fc_layers] #op is [x1 ,x2 ...] xi is of size batchsize, 240
        op = torch.stack(op, dim = 0) #op now has shape 5, batchsize, 240
        op, _ = torch.max(op, dim = 0) #op now again has dimesnion batchsize, 240
        return op


class Generator(nn.Module):
    def __init__(self, latent_dim=100, num_labels=10, output_dim=784):
        super(Generator, self).__init__()
        self.embedding = nn.Embedding.from_pretrained(torch.tensor(np.eye(num_labels)).float(), freeze = True)
        self.fc1 = nn.Sequential(
          nn.Dropout(0.5), #dropout at the beginning

          nn.Linear(100,200),
          nn.ReLU(),
          nn.Dropout(0.5)
        ) 
        self.fc2 = nn.Sequential(
          nn.Linear(num_labels,1000),
          nn.ReLU(),
          nn.Dropout(0.5)
        )
        self.fc3 = nn.Sequential(
          nn.Linear(1200,1200),
          nn.ReLU(),
          nn.Dropout(0.5)
        )
        self.fc4 = nn.Sequential(
          nn.Linear(1200,784),
          nn.Sigmoid()
        )

                
    def forward(self,z,y): # z:batch_size,100; y:batch_size,
        op1 = self.fc1(z)
        y_1hot=self.embedding(y)
        op2 = self.fc2(y_1hot)
        op = torch.cat((op1, op2), dim = -1)
        op = self.fc3(op)
        op = self.fc4(op)

        return op


class Discriminator(nn.Module):
    def __init__(self, input_dim=784, num_labels=10, output_dim=1):
        super(Discriminator, self).__init__()
        self.embedding = nn.Embedding.from_pretrained(torch.tensor(np.eye(num_labels)).float(), freeze = True)
        self.L1 = nn.Sequential(
          nn.Dropout(0.5), #dropout at the beginning

          Maxout_layer(input_dim=input_dim, output_dim=240, pieces=5),
          nn.ReLU(),
          nn.Dropout(0.5)
        )
        self.L2 = nn.Sequential(
          Maxout_layer(input_dim=num_labels, output_dim=50, pieces=5),
          nn.ReLU(),
          nn.Dropout(0.5)
        )
        self.L3 = nn.Sequential(
          Maxout_layer(input_dim=290, output_dim=240, pieces=4),
          nn.ReLU(),
          nn.Dropout(0.5)
        )
        self.fc = nn.Sequential(
          nn.Linear(in_features=240, out_features=1),
          # nn.Sigmoid()
        )
    
    def forward(self,x,y): # x:batch_size,784; y:batch_size,
        op1 = self.L1(x)
        y_1hot=self.embedding(y)
        op2=self.L2(y_1hot)
        op=torch.cat((op1,op2),dim=-1)
        op=self.L3(op)
        op = self.fc(op)
        return op


def load_gan_data_fromnumpy(traindatapath, trainlabelspath):
    X = np.load(traindatapath)
    labels = np.load(trainlabelspath)
    print('input label shape and X shape  = ', labels.shape, X.shape)

    X = -1*((X)/255. -1.) # for normalizing and making it black images

    data_Y=torch.from_numpy(labels[:X.shape[0]].astype(int))
    data_X=torch.from_numpy(X.reshape(-1, 28*28))

    #shuffle data_X and data_Y
    shuffler = np.random.permutation(data_X.shape[0])
    data_X_shuff = data_X[shuffler]
    data_Y_shuff = data_Y[shuffler]

    print('data loaded')
    print('data_X = ', data_X)
    print('data_Y = ', data_Y)
    print('data_X shape = ', data_X.shape)
    print('data_Y shape = ', data_Y.shape)

    return data_X_shuff, data_Y_shuff


def get_params_models(data_X, data_Y):
    batchsize = 100
    numclasses = torch.unique(data_Y).shape[0]

    # print(data_X.shape, data_Y.shape)
    dataset=TensorDataset(data_X ,data_Y)
    data_loader=DataLoader(dataset, batch_size=batchsize, shuffle=True, num_workers=2)

    discriminator=Discriminator(input_dim=784, num_labels = numclasses, output_dim=1)
    generator=Generator(latent_dim=100, num_labels = numclasses, output_dim=784)

    # loss_fn = nn.BCELoss() # or may use MSE
    loss_fn = nn.BCEWithLogitsLoss()

    optim_disc=optim.Adam(discriminator.parameters(), lr=0.00002)
    optim_gen=optim.Adam(generator.parameters(), lr=0.00002)

    # optim_disc=optim.SGD(discriminator.parameters(),lr=0.0001,momentum=0.5)
    # scheduler_disc=optim.lr_scheduler.ExponentialLR(optim_disc, 1/1.00004)
    # optim_gen=optim.SGD(generator.parameters(),lr=0.0001,momentum=0.5)
    # scheduler_gen=optim.lr_scheduler.ExponentialLR(optim_gen, 1/1.00004)

    generator=generator.to(device)
    discriminator=discriminator.to(device)

    return generator, discriminator, optim_gen, optim_disc, loss_fn, data_loader, numclasses

def train_cgan(generator, discriminator, optim_gen, optim_disc, loss_fn, data_loader, numclasses, root_path_to_save, num_epochs = 100):
    genlosslist = []
    dislosslist = []
    num_epochs = num_epochs
    for ep in range(num_epochs):
        # ep+=1
        disc_loss, gen_loss,= 0, 0
        for batch, (X,Y) in enumerate(data_loader):

            X,Y = X.to(device).float(), Y.to(device)
            z=torch.rand(X.shape[0],100).to(device)
            z_labels=torch.randint(low=0,high=numclasses,size=(X.shape[0],)).to(device)

            # train the discriminator
            # for i in range(5):
            optim_disc.zero_grad()
            y_real=torch.ones(X.shape[0],1).to(device)
            y_pred_real=discriminator(X,Y.long())
            y_fake=torch.zeros(X.shape[0],1).to(device)
            

            X_fake=generator(z,z_labels)
            y_pred_fake=discriminator(X_fake,z_labels)
            loss1=loss_fn(y_pred_real,y_real)
            loss2=loss_fn(y_pred_fake,y_fake)
            loss=(loss1+loss2)/2
            disc_loss+=loss.item()

            loss.backward()
            optim_disc.step()
            
            # for i in range(5):
            # train the generator
            optim_gen.zero_grad()
            y_fool=torch.ones(X.shape[0],1).to(device)
            x_fake=generator(z,z_labels)
            y_pred=discriminator(x_fake,z_labels)
            loss=loss_fn(y_pred,y_fool)
            gen_loss+=loss.item()
            loss.backward()
            optim_gen.step()

            # scheduler_disc.step()
            # scheduler_gen.step()

            '''show images generated and real'''
            if(batch%500 == 0):
                print("epoch:",ep,"discriminator loss:",disc_loss/(batch+1),"generator loss:",gen_loss/(batch+1), "lr(gen and dis) = {}".format(optim_gen.param_groups[0]['lr']))
            #     # sample_img_grid(generator, device, numclasses)
            #     sample_imgs(generator, device, numclasses)

            #     plt.figure(figsize=(15.,30.))
            #     for ind in range(10):
            #         plt.subplot(1,10,ind+1)
            #         plt.title(Y[ind].item())
            #         plt.imshow(X[ind].to('cpu').numpy().reshape((28,28)),cmap='gray')
            #         plt.axis('off')
            #     plt.show()


        genlosslist.append(gen_loss/(batch+1))
        dislosslist.append(disc_loss/(batch+1))

        #save plots
        saveplots(genlosslist, dislosslist, root_path_to_save)

        #generate some data from each class and save, to see the generated images
        sample_img_grid(generator, device, numclasses, root_path_to_save)

        print("epoch:",ep,"discriminator loss:",disc_loss/(batch+1),"generator loss:",gen_loss/(batch+1), "lr(gen and dis) = {}".format(optim_gen.param_groups[0]['lr']))


    return genlosslist, dislosslist

def saveplots(genlist, dislist, img_savepath):
    plt.figure(figsize = (10, 10))
    plt.plot(genlist, label = "gen-loss")
    plt.plot(dislist, label = "dis-loss")
    plt.xlabel('training-steps')
    plt.ylabel('Loss')
    plt.savefig(os.path.join(img_savepath, "GAN_Loss.jpg"), dpi = 100)
    plt.close()


if __name__ == "__main__":

    torch.manual_seed(0)

    parser = argparse.ArgumentParser()
    # data path for training
    parser.add_argument('--traindatapath', type=str, default = None)
    parser.add_argument('--trainlabelspath', type=str, default = None)
    #data path for pretrained generator model for testing/generating 
    parser.add_argument('--gen_model_pretr', type=str, default = None)
    #training or testigng?
    parser.add_argument('--train_or_gen', type=str, default = "train")
    #number of epochs, default is 100
    parser.add_argument('--num_epochs', type=int, default = 100)
    #for saving genmodel, dismodel and plots from training
    parser.add_argument('--root_path_to_save', type=str)

    args=parser.parse_args()

    if not os.path.exists(args.root_path_to_save):
        os.makedirs(args.root_path_to_save)

    if(args.train_or_gen == "train"):

        device='cuda:0' if torch.cuda.is_available() else 'cpu'

        data_X, data_Y = load_gan_data_fromnumpy(args.traindatapath, args.trainlabelspath)

        generator, discriminator, optim_gen, optim_disc, loss_fn, data_loader, numclasses = get_params_models(data_X, data_Y)

        genlosslist, dislosslist = train_cgan(generator, discriminator, optim_gen, optim_disc, loss_fn, data_loader, numclasses, args.root_path_to_save, num_epochs = args.num_epochs)

        # save generator and discriminator
        torch.save(generator, os.path.join(args.root_path_to_save, "gen_trained.pth"))
        torch.save(discriminator, os.path.join(args.root_path_to_save, "dis_trained.pth"))

        #save plots
        saveplots(genlosslist, dislosslist, args.root_path_to_save)

        #generate some data from each class and save, to see the generated images
        sample_img_grid(generator, device, 9, args.root_path_to_save)


    elif(args.train_or_gen == "generate"):
        # generate 1000 images from each class, also print 
        # fid score between true and generated classes

        # load generator and discriminator
        generator = torch.load(args.gen_model_pretr)
        # note: DONT PUT MODEL IN EVAL MODE

        # generate_images and save
        # for i in range(9):
            #generate and save images for ith class

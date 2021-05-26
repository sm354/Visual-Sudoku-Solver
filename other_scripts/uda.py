import argparse
import numpy as np
import os
import matplotlib.pyplot as plt
import PIL.Image as Image
import torch
from sklearn.cluster import MiniBatchKMeans, KMeans
from sklearn import decomposition
from scipy.sparse import csr_matrix
import torchvision
import torch.nn as nn
from torchvision import transforms
import torch.nn.functional as F

import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import Dataset, DataLoader
from torch.autograd import Variable
import copy

from sklearn.datasets import fetch_openml
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

device='cuda:0' if torch.cuda.is_available() else 'cpu'

torch.manual_seed(0)


parser = argparse.ArgumentParser()
parser.add_argument('--query_datapath', type=str, default = None)
parser.add_argument('--target_datapath', type=str, default = None)
parser.add_argument('--supervised_datapath', type=str, default = None) # this is actually 9k data
parser.add_argument('--supervised_labels', type=str, default = None) # this is actually 9k data

parser.add_argument('--testing_query_input', type=str, default = None) # this is actually 9k data
parser.add_argument('--output_testing_query_labels', type=str, default = None) # this is actually 9k data

parser.add_argument('--output_qt_labels', type=str, default = None) # this is actually 9k data
parser.add_argument('--output_classifier', type=str, default = None)

args=parser.parse_args()

# if not os.path.exists(args.savedir):
# 	os.makedirs(args.savedir)




# *******************************************************LOADING DATA******************************************************

X_target=np.load(args.target_datapath)
X_query=np.load(args.query_datapath)

X = np.concatenate((X_query, X_target))
# X = X_target

# oneshot_data=np.load(path+'sample_images.npy')
oneshot_data=np.load(args.supervised_datapath)

print('shape of oneshot_data', oneshot_data.shape)

#applying minibatch kmeans
X = -1*((X)/255. -1.) #for making it a sparse matrix
# X = (X)/255.
print('x ki shape', X.shape)
X=X.reshape((-1,28*28)) #shape 640k, 784
x_oneshot = -1*(oneshot_data.reshape((-1, 28*28))/(255.) -1.) #shape 10, 784
# x_oneshot = oneshot_data.reshape((-1, 28*28))/(255.) #shape 10, 784

# X = np.concatenate((X, x_oneshot))
x_oneshot_target = x_oneshot #from 0th class to 8th class, 9th dropped as its no where in the images i THINK
# x_oneshot_target = x_oneshot[:-1] #from 0th class to 8th class, 9th dropped as its no where in the images i THINK


print('shape of X', X.shape)
print('shape of x_oneshot', x_oneshot.shape)
print('shape of x_oneshot_target', x_oneshot_target.shape)

print('X \n', X)
print('x_oneshot \n', x_oneshot)
print('x_oneshot_target \n', x_oneshot_target)

X = X.reshape(-1, 1, 28, 28)
print(X.shape)

class CustomTensorDataset_pair(Dataset):
    """TensorDataset with support of transforms.
    """
    def __init__(self, tensors, transform=None):
        assert all(tensors[0].size(0) == tensor.size(0) for tensor in tensors)
        self.tensors = tensors
        self.transform = transform

    def __getitem__(self, index):

        x = self.tensors[0][index]
        # print(x.shape)
        if self.transform:
            x = self.transform(x)

        y = self.tensors[1][index]

        return x, y

    def __len__(self):
        return self.tensors[0].size(0)


# we have supervised data (10) and unsuper vised data (1280000) which is X
# apply transformations on X 
# X can be first shuffled 
shuffler = np.random.permutation(X.shape[0])
X = X[shuffler]
X = torch.tensor(X)
# X = X[:9000]
X = X[:18000]
print('shape of X now after sampling for making final unsup data = ', X.shape)


#now sequentially select batches of X and apply transformations
# select transformations
# t0 =  transforms.RandomApply()
t1 = transforms.RandomRotation(20)
# t2 = transforms.RandomCrop((28, 28), padding = 4)
t2 = transforms.RandomCrop((28, 28))

t3 = transforms.RandomPerspective()
trans = transforms.Compose([transforms.ToPILImage(), t1, t2, t3, transforms.ToTensor()])

unsup_dataset = CustomTensorDataset_pair(tensors = (X.float(), X), transform=trans)
unsup_train_loader = torch.utils.data.DataLoader(unsup_dataset, batch_size=180)




#making supervised dataset ----  unsupervised is already made above
sup_onsht_data = torch.tensor(x_oneshot_target.reshape(-1, 1, 28, 28))
# sup_onsht_labels = torch.tensor([i for i in range(9)])
sup_onsht_labels = torch.tensor(np.load(args.supervised_labels))

shuffler = np.random.permutation(sup_onsht_data.shape[0])
sup_onsht_data = sup_onsht_data[shuffler]
sup_onsht_labels = sup_onsht_labels[shuffler]

print(sup_onsht_labels, sup_onsht_labels.shape)
print('supervised datashape = ', sup_onsht_data.shape)



# sup_dataset = CustomTensorDataset(tensors = sup_onsht_data)
num_batches = len(unsup_train_loader)
# sup_data = torch.cat([sup_onsht_data for i in range(num_batches)], dim = 0)
# sup_labels = torch.cat([sup_onsht_labels for i in range(num_batches)], dim = 0)

sup_data = sup_onsht_data
sup_labels = sup_onsht_labels

print(sup_data.shape)

sup_dataset = CustomTensorDataset_pair(tensors = (sup_data.float(), sup_labels), transform=trans)
# sup_dataset = CustomTensorDataset_pair(tensors = (sup_data, sup_labels))
sup_train_loader = torch.utils.data.DataLoader(sup_dataset, batch_size = 90, shuffle = False)

print(len(sup_train_loader))



print('sup and unsup trainloader shape = ', len(sup_train_loader), len(unsup_train_loader))


X_target=np.load(args.target_datapath)
X = X_target
X = -1*((X)/255. -1.) #for making it a sparse matrix

print('x ki shape', X.shape)
X=X.reshape((-1,28*28)) #shape 640k, 784

print('Xtarget shape', X)

batchsize = 128
target_loader = DataLoader(X.reshape(-1, 1, 28, 28), batch_size=batchsize, shuffle=False)


def predict(model, device, test_loader, use_cuda):
    model.eval()
    predictions = []
    with torch.no_grad():
        for data in test_loader:
            data = data.to(device)
            output = model(data.float())
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            predictions.extend(pred.tolist())

    # print(predictions)

    return np.array(predictions)


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
    # print(labels.shape)
    # print(labels[0])
    # print(labels[10000])

    subatomic_correct = 0

    correct = 0
    total = 0
    #now we have labels of correct shape
    final_bool_arr = np.array([True for i in range(labels.shape[0])])
    for i in range(8):
        k = i * 2 if i<4 else (i-4) * 2
        j= (i // 4) * 4
        # print(k, j)
        # if(np.all(np.apply_along_axis(is_set_correct, axis = 1, arr = labels[:, :, i])) == True or np.all(np.apply_along_axis(is_set_correct, axis = 1, arr = labels[:, i, :])) == True or np.all(np.apply_along_axis(is_set_correct, axis = 1, arr = labels[:, k:k+2, j:j+4].reshape(-1, 8))) !=True ):
        #   correct+=1
        # total+=1

        arr1 = np.apply_along_axis(is_set_correct, axis = 1, arr = labels[:, :, i])
        arr2 = np.apply_along_axis(is_set_correct, axis = 1, arr = labels[:, i, :])
        arr3 = np.apply_along_axis(is_set_correct, axis = 1, arr = labels[:, k:k+2, j:j+4].reshape(-1, 8))
        arr = arr1*arr2*arr3
        # arr = arr1*arr2
        assert(arr.shape[0] == labels.shape[0] and len(arr.shape) == 1)
        final_bool_arr *= arr
        subatomic_correct += arr1.sum() + arr2.sum() + arr3.sum()
        # subatomic_correct += arr1.sum() + arr2.sum()

    return final_bool_arr.sum()/final_bool_arr.shape[0], subatomic_correct/(3*8*labels.shape[0])


# classifier network
class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 5, padding = 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1   = nn.Linear(400, 120)
        self.fc2   = nn.Linear(120, 84)
        self.fc3   = nn.Linear(84, 9)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, (2, 2))
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, (2, 2))
        x = x.view(-1, np.prod(x.size()[1:]))
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.fc3(x)
        return x


model = LeNet().to(device)


test_batch_size=1000
epochs=25
lr=0.1
gamma=0.987
no_cuda=False
seed=1
log_interval=100
save_model=False
use_cuda = not no_cuda and torch.cuda.is_available()
torch.manual_seed(seed)
device = torch.device("cuda" if use_cuda else "cpu")
optimizer = optim.Adam(model.parameters(), lr=0.0002)
scheduler = StepLR(optimizer, step_size=1, gamma=gamma)



for epoch in range(epochs):
    model.train()
    acc = 0
    for batch_idx, (Y, X) in enumerate(zip(unsup_train_loader, sup_train_loader)):

        (Xtrans, Xnotrans)= Y
        (Xsup, labels) = X

        Xtrans, Xnotrans, Xsup, labels = Xtrans.to(device), Xnotrans.to(device), Xsup.to(device), labels.to(device)
        optimizer.zero_grad()

        # print(Xtrans.shape, Xnotrans.shape, Xsup.shape, labels.shape)

        softmax = nn.Softmax(dim=1)
        temp_model = copy.deepcopy(model).eval()

        sup_out = model(Xsup.float())
        with torch.no_grad():
            unsup_notrans_out = softmax(temp_model(Xnotrans.float()))
        unsup_trans_out = softmax(model(Xtrans.float()))

        loss_sup = nn.CrossEntropyLoss()
        loss_unsup = nn.BCELoss()

        l2unsup = loss_unsup(unsup_trans_out, unsup_notrans_out)
        l1sup = loss_sup(sup_out, labels.long())
        total_loss = (l2unsup+ 10*l1sup)

        acc += (torch.argmax(sup_out, dim=1).long() == labels.long()).sum().item()/(labels.shape[0])

        total_loss.backward()
        optimizer.step()


    print('epoch = {}, loss1sup = {}, loss2usup = {}, acc = {}'.format(epoch, l1sup.item(), l2unsup.item(), acc/(batch_idx+1)))

    if(epoch% 5 == 0):
        target_labels = predict(model, device, target_loader, True)
        print(clustering_accuracy(target_labels))


torch.save(model, args.output_classifier)


#classify query+target images and save

X_target=np.load(args.target_datapath)
X_query=np.load(args.query_datapath)

X = np.concatenate((X_query, X_target))

X = -1*((X)/255. -1.) #for making it a sparse matrix
print('x ki shape', X.shape)
X=X.reshape((-1,28*28)) #shape 640k, 784


model.eval()
# targetset = TensorDataset(X[40000:] ,data_Y[40000:])
batchsize = 128
data_loader = DataLoader(X.reshape(-1, 1, 28, 28), batch_size=batchsize, shuffle=False)

def predict(model, device, test_loader, use_cuda):
    model.eval()
    predictions = []
    with torch.no_grad():
        for data in test_loader:
            data = data.to(device)
            output = model(data.float())
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            predictions.extend(pred.tolist())

    # print(predictions)

    return np.array(predictions)



data_labels = predict(model, device, data_loader, True)
data_labels.shape

#save labels of query and target
np.save(args.output_qt_labels, data_labels)






#TESTING QUERY
X=[] #this will contain 28,28 images

# i = 0
for img_name in sorted(os.listdir(args.testing_query_input)):
    # i+=1
    # if(i ==3):
    #     break
    img = np.array(Image.open(os.path.join(args.testing_query_input,img_name))) # 224,224 = 64 * 28,28
    sub_imgs=np.split(img,8)
    sub_imgs=[np.split(x_,8,axis=1) for x_ in sub_imgs]
    sub_imgs=np.array(sub_imgs) # 8,8,28,28
    sub_imgs=sub_imgs.reshape((-1,28,28))
    X.append(sub_imgs)

X=np.array(X)
X_input_query=X.reshape((-1,28,28))


X_input_query = -1*((X_input_query)/255. -1.) #for making it a sparse matrix


batchsize = 128
data_loader = DataLoader(X_input_query.reshape(-1, 1, 28, 28), batch_size=batchsize, shuffle=False)

def predict(model, device, test_loader, use_cuda):
    model.eval()
    predictions = []
    with torch.no_grad():
        for data in test_loader:
            data = data.to(device)
            output = model(data.float())
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            predictions.extend(pred.tolist())

    # print(predictions)

    return np.array(predictions)



data_labels = predict(model, device, data_loader, True)
print(data_labels.shape)

#save labels of query and target
np.save(args.output_testing_query_labels, data_labels)





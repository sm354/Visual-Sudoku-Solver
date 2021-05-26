# import numpy as np
# import os
# import matplotlib.pyplot as plt
# import PIL.Image as Image
# import torch
# from sklearn.cluster import MiniBatchKMeans, KMeans
# from sklearn import decomposition
# from scipy.sparse import csr_matrix
# import torchvision
# import torch.nn as nn
# from torchvision import transforms
# import torch.nn.functional as F

# import torch.optim as optim
# from torchvision import datasets, transforms
# from torch.optim.lr_scheduler import StepLR
# from torch.utils.data import Dataset, DataLoader, TensorDataset
# from torch.autograd import Variable
# import argparse

# # classifier network
# class LeNet(nn.Module):
#     def __init__(self, num_classes):
#         super(LeNet, self).__init__()
#         self.conv1 = nn.Conv2d(1, 6, 5, padding = 2)
#         self.conv2 = nn.Conv2d(6, 16, 5)
#         self.fc1   = nn.Linear(400, 120)
#         self.fc2   = nn.Linear(120, 84)
#         self.fc3   = nn.Linear(84, num_classes)

#     def forward(self, x):
#         x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
#         x = F.max_pool2d(F.relu(self.conv2(x)), (2, 2))
#         x = x.view(-1, np.prod(x.size()[1:]))
#         x = F.relu(self.fc1(x))
#         x = F.relu(self.fc2(x))
#         x = self.fc3(x)
#         return x

# # train classifier 
# def train(log_interval, model, device, train_loader, optimizer, epoch,use_cuda):
#     model.train()
#     for batch_idx, (data, target) in enumerate(train_loader):
#         data, target = data.to(device), target.to(device)
#         optimizer.zero_grad()
#         output = model(data.float())
#         loss = nn.CrossEntropyLoss()
#         target = torch.tensor(target, dtype=torch.long, device=device)
#         lo=loss(output, target)
#         lo.backward()
#         optimizer.step()
#         if batch_idx % log_interval == 0:
#             print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
#                 epoch, batch_idx * len(data), len(train_loader.dataset),
#                 100. * batch_idx / len(train_loader), lo.item()))


# def test(model, device, test_loader,use_cuda):
#     model.eval()
#     test_loss = 0
#     correct = 0
#     with torch.no_grad():
#         for data, target in test_loader:
#             data, target = data.to(device), target.to(device)
#             output = model(data.float())
#             loss = nn.CrossEntropyLoss()
#             target = torch.tensor(target, dtype=torch.long, device=device)
#             lo=loss(output, target)
#             test_loss +=lo.item()  # sum up batch loss
#             pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
#             correct += pred.eq(target.view_as(pred)).sum().item()
#             data, target = data.to("cpu"), target.to("cpu")

#     test_loss /= len(test_loader.dataset)

#     print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
#         test_loss, correct, len(test_loader.dataset),
#         100. * correct / len(test_loader.dataset)))

#     return test_loss,correct / len(test_loader.dataset)

# def main(num_classes):
#     test_batch_size=1000
#     epochs=2
#     lr=0.1
#     gamma=0.987
#     no_cuda=False
#     seed=1
#     log_interval=100
#     save_model=False

#     use_cuda = not no_cuda and torch.cuda.is_available()

#     torch.manual_seed(seed)

#     device = torch.device("cuda:0" if use_cuda else "cpu")

#     model = LeNet(num_classes).to(device)
#     # if use_cuda:
#     #     model=torch.nn.DataParallel(model)
#     #     torch.backends.cudnn.benchmark=True
#     optimizer = optim.SGD(model.parameters(), lr=lr)
#     l1=[0]*epochs
#     l2=[0]*epochs
#     l3=[0]*epochs
#     l4=[0]*epochs
#     scheduler = StepLR(optimizer, step_size=1, gamma=gamma)
#     for epoch in range(1, epochs + 1):
#         train(log_interval, model, device, train_loader, optimizer, epoch,use_cuda)

#         l1[epoch-1],l2[epoch-1]=test(model, device, test_loader, use_cuda)
#         l3[epoch-1],l4[epoch-1]=test(model, device, train_loader,use_cuda)
#         scheduler.step()

#     return model

# def load_gan_data_fromnumpy(traindatapath, trainlabelspath):
#     X = np.load(traindatapath)
#     labels = np.load(trainlabelspath)

#     print(traindatapath, X)

#     X = (X)/255.

#     data_Y=torch.from_numpy(labels.astype(int))
#     data_X=torch.from_numpy(X.reshape(-1, 1, 28, 28))

#     #shuffle data_X and data_Y
#     shuffler = np.random.permutation(data_X.shape[0])
#     data_X_shuff = data_X[shuffler]
#     data_Y_shuff = data_Y[shuffler]

#     print('data loaded')
#     print('data_X = ', data_X)
#     print('data_Y = ', data_Y)
#     print('data_X shape = ', data_X.shape)
#     print('data_Y shape = ', data_Y.shape)

#     return data_X_shuff, data_Y_shuff

#     model.eval()
#     predictions = []
#     with torch.no_grad():
#         for data in test_loader:
#             data = data.to(device)
#             output = model(data.float())
#             pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
#             predictions.extend(pred.tolist())

#     # print(predictions)

#     return np.array(predictions)

# def predict(model, device, test_loader, use_cuda):
#     model.eval()
#     predictions = []
#     with torch.no_grad():
#         for data in test_loader:
#             data = data.to(device)
#             output = model(data.float())
#             pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
#             predictions.extend(pred.tolist())

#     # print(predictions)

#     return np.array(predictions)

# def is_set_correct(array):
#     # print(array)
#     # print(set(array))
#     if len(set(array)) >= 8:
#         return True
#     return False


# def clustering_accuracy(labels):
#     #labels are of shape (totalsmall images in all sudoku which is divisible by 64,)
#     labels = labels.reshape((labels.shape[0]//64, -1))
#     labels = labels.reshape((-1, 8, 8))
#     print(labels.shape)
#     print(labels[0])
#     # print(labels[10000])

#     subatomic_correct = 0

#     correct = 0
#     total = 0
#     #now we have labels of correct shape
#     final_bool_arr = np.array([True for i in range(labels.shape[0])])
#     for i in range(8):
#         k = i * 2 if i<4 else (i-4) * 2
#         j= (i // 4) * 4
#         print(k, j)
#         # if(np.all(np.apply_along_axis(is_set_correct, axis = 1, arr = labels[:, :, i])) == True or np.all(np.apply_along_axis(is_set_correct, axis = 1, arr = labels[:, i, :])) == True or np.all(np.apply_along_axis(is_set_correct, axis = 1, arr = labels[:, k:k+2, j:j+4].reshape(-1, 8))) !=True ):
#         #   correct+=1
#         # total+=1

#         arr1 = np.apply_along_axis(is_set_correct, axis = 1, arr = labels[:, :, i])
#         arr2 = np.apply_along_axis(is_set_correct, axis = 1, arr = labels[:, i, :])
#         arr3 = np.apply_along_axis(is_set_correct, axis = 1, arr = labels[:, k:k+2, j:j+4].reshape(-1, 8))
#         arr = arr1*arr2*arr3
#         # arr = arr1*arr2
#         assert(arr.shape[0] == labels.shape[0] and len(arr.shape) == 1)
#         final_bool_arr *= arr
#         subatomic_correct += arr1.sum() + arr2.sum() + arr3.sum()
#         # subatomic_correct += arr1.sum() + arr2.sum()

#     return final_bool_arr.sum()/final_bool_arr.shape[0], subatomic_correct/(3*8*labels.shape[0])



# if __name__ == "__main__":
#     torch.manual_seed(0)
#     device='cuda:0' if torch.cuda.is_available() else 'cpu'

#     parser = argparse.ArgumentParser()
#     # data path for training
#     parser.add_argument('--traindatapath', type=str, default = None)
#     parser.add_argument('--trainlabelspath', type=str, default = None)
#     #number of epochs
#     parser.add_argument('--num_classes', type=int, default = 9)
#     #for saving classifier model from training
#     parser.add_argument('--root_path_to_save', type=str)
#     #target datapath for testing our classifier
#     parser.add_argument('--targetdatapath', type=str)
#     args=parser.parse_args()

#     if not os.path.exists(args.root_path_to_save):
#         os.makedirs(args.root_path_to_save)

#     data_X_shuff, data_Y_shuff = load_gan_data_fromnumpy(args.traindatapath, args.trainlabelspath)

#     total_points = data_X_shuff.shape[0]
#     batchsize = 128
#     trainset = TensorDataset(data_X_shuff[0:int(total_points*4//5)] ,data_Y_shuff[0:int(total_points*4//5)])
#     train_loader = DataLoader(trainset, batch_size=batchsize, shuffle=True)

#     testset = TensorDataset(data_X_shuff[int(total_points*4//5):int(total_points*4.5//5)] ,data_Y_shuff[int(total_points*4//5):int(total_points*4.5//5)])
#     test_loader = DataLoader(testset, batch_size=batchsize, shuffle=True)

#     test_final = TensorDataset(data_X_shuff[int(total_points*4.5//5):total_points] ,data_Y_shuff[int(total_points*4.5//5):total_points])
#     test_final_loader = DataLoader(test_final, batch_size=batchsize, shuffle=True)

#     print("length of dataloaders = ", len(train_loader), len(test_loader), len(test_final_loader))

#     model=main(args.num_classes)

#     #save classifier model
#     torch.save(model, os.path.join(args.root_path_to_save, "classifier_trained.pth"))


#     print("____________Performance of trained classifiier on target set sudoku____________")
#     classifier = torch.load(os.path.join(args.root_path_to_save, "classifier_trained.pth"))
#     classifier.eval()

#     #load target dataset data
#     Xtarget = np.load(args.targetdatapath)
#     Xtarget = Xtarget/255. # for normalizing and making it black images
#     Xtarget=torch.from_numpy(Xtarget.reshape(-1, 1, 28, 28))
#     batchsize = 128
#     target_loader = DataLoader(Xtarget, batch_size=batchsize, shuffle=False)

#     target_labels = predict(model, device, target_loader, True)
#     print('target_labels shape = ', target_labels.shape)

#     print('clustering_performance = ', clustering_accuracy(target_labels))

    




import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader,random_split
import torch.optim as optim
import cv2
import time
import sys
import gc
import numpy as np
from sklearn.metrics import classification_report
import os 
from pathlib import Path

__author__ = "Alberto Abarzua"


""" 
  
  
  Implementation of DenseNet-121  with 4 dense blocks. changing the input size to be 32,32 3 channel images.

  Original paper:
    https://arxiv.org/abs/1608.06993

      
"""

class DenseModule(nn.Module):
  """Implementation of Dense Block used in DenseNet"""
  def __init__(self, in_chanels,k, N):
    super(DenseModule, self).__init__()
    self.k = k
    self.n =N
    self.in_chanels = in_chanels
    self.convs_1x1 =nn.ModuleList([])
    self.convs_3x3 =nn.ModuleList([])
    self.relu = nn.ReLU()
    self.bns = nn.ModuleList([])
    for i in range(self.n):
      self.convs_1x1.append(nn.Conv2d(self.in_chanels + i*(self.k),k,(1,1),stride =1,padding = 0))
      self.convs_3x3.append(nn.Conv2d(k,k,(3,3),stride =1,padding =1))
      self.bns.append(nn.BatchNorm2d(self.in_chanels + (i+1)*(self.k),track_running_stats =True))
    
  def forward(self, x):
    cur_x = x
    for i in range(self.n):
      
      cur_val = self.convs_1x1[i](cur_x)
      cur_val = self.convs_3x3[i](cur_val)
      cur_x = torch.cat([cur_x,cur_val],dim=1)
      #cur_x = self.bns[i](cur_x)
      cur_x = self.relu(cur_x)
      cur_val = cur_x
    return cur_x
      
class TransitionModule(nn.Module):
  """Implementation of Transition Layer used in DenseNet"""
  def __init__(self, in_chanels,out_channels,conv_kernel = (2,2),pooling_size = (2,2),pool_type = "avg",stride_c =1,stride_p =2,padding_c =0,padding_p=0):
      super(TransitionModule, self).__init__()
      self.in_chanels = in_chanels
      self.conv = nn.Conv2d(self.in_chanels,out_channels,conv_kernel,stride = stride_c,padding = padding_c)
      self.bn = nn.BatchNorm2d(out_channels,)
      if (pool_type == "avg"):
        self.pooling = nn.AvgPool2d(pooling_size,stride = stride_p,padding = padding_p)
      else:
        self.pooling = nn.MaxPool2d(pooling_size,stride = stride_p,padding = padding_p)
      

  def forward(self, x):
    x = self.conv(x)
    x = self.bn(x)
    x = F.relu(x)
    x = self.pooling(x)
    return x





class DenseNet(nn.Module):
  """Adaptation of DenseNet-121"""
  def __init__(self, n_classes):
    super(DenseNet, self).__init__()
    self.k = 32
    self.t1 = TransitionModule(3,2*self.k,(7,7),(3,3),"max",stride_c =1,stride_p=1,padding_c = 2,padding_p =1)
    self.d1 = DenseModule(2*self.k,self.k,6)
    
    self.t2 = TransitionModule(self.k*8,self.k*4,(1,1),(3,3),"avg",stride_c =1,stride_p=1)
    self.d2 = DenseModule(4*self.k,self.k,12)
    self.t3= TransitionModule(self.k*16,self.k*8,(1,1),(2,2),"avg")
    self.d3 = DenseModule(8*self.k,self.k,24)
    self.t4 = TransitionModule(32*self.k,self.k*16,(1,1),(2,2),"avg")
    self.d4 = DenseModule(16*self.k,self.k,16)
    self.average = nn.AvgPool2d((7,7),stride=1)

    self.fc_out = nn.Linear(32*self.k, n_classes)

    self.softmax = torch.nn.Softmax(dim =1)

  def forward(self, x):
    """Calculates the foward operation of the net (does not apply softmax)

    Args:
        x (torch.tensor): tensor of dimension (BATCH,3,32,32) containing the images to predict

    Returns:
        torch.tensor: tensor of dimension (BATCH,4)containing the predictions (does not apply softmax)
    """
    x = self.t1(x)
    x = self.d1(x)
    x = self.t2(x)
    x = self.d2(x)
    x = self.t3(x)
    x = self.d3(x)
    x = self.t4(x)
    x = self.d4(x)
    x = self.average(x)

    # N x out_size
    logits = self.fc_out(x.view(-1,32*self.k))

    return logits


  def predict_proba(self,x):
    """Returns the foward pass of the net appying softmax to the output

    Args:
        x (_type_): _description_

    Returns:
        _type_: _description_
    """
    return self.softmax(self.forward(x))


def apply_transforms(transforms,x):
      imgs = torch.zeros((len(x),3,32,32))
      for i in range(len(x)):
          cur = x[i]
          for tr in transforms:
              cur  = tr(cur)
          imgs[i]+= cur
      return imgs



def train_for_classification(net, train_loader, test_loader, optimizer, criterion, lr_scheduler,  epochs, device):
  """Function used to train this implementation of DenseNet

  Args:
      net (DenseNet): model to train
      train_loader (DataLoader): dataloader with the train dataset
      test_loader (DataLoader): dataloader with the test dataset
      optimizer (Optimizer): Optimizer used to train de model's parameters
      criterion (Criterion): criterion used to calculate the loss during training
      lr_scheduler (scheduler): scheduler used to control learning rate
      epochs (int): number of epochs to train
      device (string): device where the model will be trained ("cuda" or "cpu")

  Returns:
      tuple(list[float],tuple(list[float],list[float])): tuple with list of values representing (train_loss,(train_acc,test_acc))
  """

  net.to(device)
  total_train = len(train_loader.dataset)
  total_test = len(test_loader.dataset)
  epoch_duration = 0
  reports_every = 1 

  train_loss, train_acc, test_acc = [], [], []

  for e in range(1,epochs+1):  
    #Inital values.
    running_loss, running_acc = 0.0, 0.0
    epoch_start = time.perf_counter()

    net.train()
    for i, data in enumerate(train_loader):
      X, Y = data
      X, Y = X.to(device), Y.to(device)

      optimizer.zero_grad()

      Y_pred = net(X) #(BatchSize,num_classes)
      loss = criterion(Y_pred, Y)

      loss.backward()
      optimizer.step()
      # loss
      items = min(total_train, (i+1) * train_loader.batch_size)
      running_loss += loss.item()
      avg_loss = running_loss/(i+1)
      
      # accuracy
      _, max_idx = torch.max(Y_pred, dim=1)
      running_acc += torch.sum(max_idx == Y).item()
      avg_acc = running_acc/items*100
      # report
      print(f"Epoch:{e}({items}/{total_train}) lr:{lr_scheduler.get_last_lr()[0]:02.7f}," ,
          f"Loss:{avg_loss:02.5f},Train Accuracy:{avg_acc:02.1f}%'",end = "\r")
      #Memory optimization
      gc.collect()
      torch.cuda.empty_cache()


    epoch_duration += time.perf_counter() - epoch_start
    if e % reports_every == 0:
      print('Validating...',end = "\r")
      train_loss.append(avg_loss)
      train_acc.append(avg_acc)

      net.eval()
      with torch.no_grad():
        running_acc = 0.0
        for i, data in enumerate(test_loader):
          X, Y = data
          X, Y = X.to(device), Y.to(device)
          Y_pred = net(X)
          _, max_idx = torch.max(Y_pred, dim=1)
          running_acc += torch.sum(max_idx == Y).item()
          avg_acc = running_acc/total_test*100
        test_acc.append(avg_acc)
        print(f"Validation acc :{avg_acc:02.2f}%  Avg-Time: {epoch_duration/e:.2f}s.")
    else:
      print()

    if lr_scheduler is not None:
      lr_scheduler.step()

  return train_loss, (train_acc, test_acc)


class ObjectLoader(Dataset):
  """Dataset class used to represent datasets creatend using "create_dataset.ipynb", with the format 
    np.array of (img,label)

  """
  def __init__(self,data,transforms):
      super(ObjectLoader,self).__init__()
      self.data = data.copy()
      self.labels = data[:,1].copy()
      self.imgs = data[:,0].copy()
      self.transforms = transforms
      self.lables_list =list(set(data[:,1]))
      self.label_dict = {elem:i for i,elem in enumerate(self.lables_list)}
      self.transform_data()

  def transform_data(self):
    """ Applies the transforms defined in init to the images, one by one."""
    prev = self.imgs.copy()
    self.imgs = torch.zeros((len(prev),3,32,32))
    for i in range(len(prev)):
        cur = prev[i]
        temp =self.labels[i] 
        self.labels[i] = self.label_dict[temp]
        for tr in self.transforms:
            cur  = tr(cur)
        self.imgs[i]+= cur
      


  def __getitem__(self,idx):
    """Gets a pair (img,label)

    Args:
        idx (int): index of the elemnt to get

    Returns:
        tuple(torch.tensor(3,32,32),torch.tensor(num_labels)): tuple with value and label in one hot encoded tensor.
    """
    img ,label= self.imgs[idx], self.labels[idx]
    return img,label


  def __len__(self):
    """Gets the length of the dataset (number of elements)

    Returns:
        int: len of data
    """
    return len(self.imgs)


#Paths
BASE = Path(__file__).parent.joinpath("data","dataset","output_data")
OUT = BASE / "densenet"
DATASET = BASE / "np_data.npy"

if __name__ =="__main__":

    #LOADING DATA AND CREATING DATASET
    from PIL import Image
    data = np.load(DATASET,allow_pickle= True)
    trs= [lambda x: cv2.resize(x,(32,32),interpolation= cv2.INTER_AREA ),lambda x : Image.fromarray(np.uint8(x)).convert('RGB'),transforms.ToTensor(),transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    dataset = ObjectLoader(data,trs)
    data_size = len(dataset)
    train_size = int(data_size*0.8)
    test_size = data_size - train_size
    trainset,testset = random_split(dataset,[train_size,test_size])
    print("Number of elements in dataset :",data_size)
    print(f"Training with: {train_size} and testing with: {test_size}")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Starting training using : {device.__str__()}")
    # Training params

    BATCH_SIZE = 32
    EPOCHS = 6
    REPORTS_EVERY = 1

    denseNet = DenseNet(len(dataset.lables_list)) 
    denseOpt = optim.Adam(denseNet.parameters(),lr=0.005) 
    criterion = nn.CrossEntropyLoss() 
    scheduler = optim.lr_scheduler.StepLR(denseOpt, step_size=2, gamma=0.5)
    train_loader = DataLoader(trainset, batch_size=BATCH_SIZE,
                              shuffle=True)
    test_loader = DataLoader(testset, batch_size=4*BATCH_SIZE,
                            shuffle=False)

    train_loss1, acc1 = train_for_classification(denseNet, train_loader, 
                                              test_loader, denseOpt, 
                                              criterion, lr_scheduler=scheduler, 
                                              epochs=EPOCHS,device=device)

    print("Saving the model's state dict")
 


    dense_data = {
    "model_state": denseNet.state_dict(),
    "labels": dataset.label_dict
            }

    torch.save(dense_data, OUT)

    
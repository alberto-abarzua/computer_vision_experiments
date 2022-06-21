import torch
import torch.nn as nn
import torch.nn.functional as F

class DenseNet(nn.Module):
  def __init__(self, n_classes):
    super(DenseNet, self).__init__()
    self.k = 32
    # Define las capas de convolución y pooling de tu arquitectura
    #pdb.set_trace()
    self.t1 = TransitionModule(3,2*self.k,(7,7),(3,3),"max",stride_c =1,stride_p=1,padding_c = 2,padding_p =1)
    self.d1 = DenseModule(2*self.k,self.k,6)
    
    self.t2 = TransitionModule(self.k*8,self.k*4,(1,1),(3,3),"avg",stride_c =1,stride_p=1)
    self.d2 = DenseModule(4*self.k,self.k,12)
    self.t3= TransitionModule(self.k*16,self.k*8,(1,1),(2,2),"avg")
    self.d3 = DenseModule(8*self.k,self.k,24)
    self.t4 = TransitionModule(32*self.k,self.k*16,(1,1),(2,2),"avg")
    self.d4 = DenseModule(16*self.k,self.k,16)
    self.average = nn.AvgPool2d((7,7),stride=1)

    # Capa de salida (antes de la función de salida)
    self.fc_out = nn.Linear(32*self.k, n_classes)

  def forward(self, x):
    # Computa las representaciones internas de la red
    
    x = self.t1(x)
    x = self.d1(x)
    x = self.t2(x)
    x = self.d2(x)
    x = self.t3(x)
    hidden = x.view(-1,256*14*14)#dimensiones (B,256,14,14)
    x = self.d3(x)
    x = self.t4(x)
    x = self.d4(x)
    x = self.average(x)

    # N x out_size
    logits = self.fc_out(x.view(-1,32*self.k))

    # En hidden debes devolver alguna de las capas oculta de la red
    return {'hidden': hidden, 'logits': logits}
# Acá el código para tu primera arquitectura

class DenseModule(nn.Module):
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
      self.bns.append(nn.BatchNorm2d(self.in_chanels + (i+1)*(self.k),track_running_stats =False))
    

  def forward(self, x):
    
    cur_x = x
    for i in range(self.n):
      
      cur_val = self.convs_1x1[i](cur_x)
      cur_val = self.convs_3x3[i](cur_val)
      #print(cur_x.size(),i)
      cur_x = torch.cat([cur_x,cur_val],dim=1)
      cur_x = self.bns[i](cur_x)
      cur_x = self.relu(cur_x)
      cur_val = cur_x
    return cur_x
      
class TransitionModule(nn.Module):
  def __init__(self, in_chanels,out_channels,conv_kernel = (2,2),pooling_size = (2,2),pool_type = "avg",stride_c =1,stride_p =2,padding_c =0,padding_p=0):
    super(TransitionModule, self).__init__()
    self.in_chanels = in_chanels
    self.conv = nn.Conv2d(self.in_chanels,out_channels,conv_kernel,stride = stride_c,padding = padding_c)
    #self.conv = nn.Conv2d(in_chanels,self.k*self.theta,conv_kernel,stride =stride)
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
import torch
import math
import numpy as np
from torch.optim import AdamW
from torch.nn.init import kaiming_uniform_
from torch.nn.init import kaiming_normal_
from torch.nn import Linear, LeakyReLU, Embedding

from btf_extractor import Ubo2014
import matplotlib.pyplot as plt
import os

import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

import time
import pickle



# model definition
class MLP(torch.nn.Module):
    # define model elements
    def __init__ (self):
        super(MLP, self).__init__()
        # input to first hidden layer
        self.H = Embedding(400, 8)
        self.D = Embedding(400, 8)
        self.U = Embedding(160000, 16)
        kaiming_uniform_(self.H.weight, nonlinearity='leaky_relu')
        kaiming_uniform_(self.D.weight, nonlinearity='leaky_relu')
        kaiming_uniform_(self.U.weight, nonlinearity='leaky_relu')
        # first hidden layer
        self.hidden1 = Linear(32, 32)
        self.act1 = LeakyReLU()

        # second hidden layer
        self.hidden2 = Linear(32, 32)
        self.act2 = LeakyReLU()

        # third hidden layer
        self.hidden3 = Linear(32, 32)
        self.act3 = LeakyReLU()

        # forth hidden layer
        self.hidden4 = Linear(32, 3)
        self.act4 = LeakyReLU()

        kaiming_uniform_(self.hidden1.weight, nonlinearity='leaky_relu')
        kaiming_uniform_(self.hidden2.weight, nonlinearity='leaky_relu')
        kaiming_uniform_(self.hidden3.weight, nonlinearity='leaky_relu')
        kaiming_uniform_(self.hidden4.weight, nonlinearity='leaky_relu')

    # forward propagate input
    def forward(self, X): # input only varies in last dimention,
        # X的最后一维必定只有3个数，第一个数是H的坐标编码，∈[0, 400), 第二个数是D的编码，∈[0, 400) ，最后则是U的编码，∈[0, 160000) 
        # 在本例中
        h = X[..., 0]
        d = X[..., 1]
        u = X[..., 2]
        # feature planes' value
        fH = self.H(h)
        fD = self.D(d)
        fU = self.U(u)

        firstLayer = torch.cat((fD, fH, fU), dim = len(X.shape) - 1) # 按照最后一维进行拼接，得到第一列方格的结果

        firstLayer = self.hidden1(firstLayer) # 第一层FC
        secondLayer = self.act1(firstLayer) # 第一层激活，得到第二列方格的结果

        secondLayer = self.hidden2(secondLayer) # 第二层FC
        thirdLayer = self.act2(secondLayer) # 第二层激活，得到第三列方格的结果

        thirdLayer = self.hidden3(thirdLayer) # 第三层FC
        forthLayer = self.act3(thirdLayer)# 第三层激活，得到第四列方格的结果

        forthLayer = self.hidden4(forthLayer) # 第四层FC
        fifthLayer = self.act4(forthLayer) # 第四层激活，得到RGB
        return fifthLayer

# 自定义数据集类
class MyDataset(Dataset):
    def __init__(self, BTFName):
        print("loading BTF:", BTFName)
        with open("dataset/Leather08/" + 'H.pkl', 'rb') as file:
            self.H = pickle.load(file)
        with open("dataset/Leather08/" + 'D.pkl', 'rb') as file:
            self.D = pickle.load(file)
        with open("dataset/Leather08/" + 'U.pkl', 'rb') as file:
            self.UQueries = pickle.load(file)
        print("initialize finished")


    def __len__(self):
        """返回数据集的大小"""
        return 151 * 151 # 该数据集一共有151 * 151张图片，每次get item返回一张图片
    
    def get_image(self, id):
        with open("dataset/Leather08/" + str(id) + '.pkl', 'rb') as file:
            image = pickle.load(file)
        return torch.tensor(image) # 这里是否需要tensor存疑，但工作最好还是交给GPU去做
        

    def __getitem__(self, idx): # 这里idx是图片编号, 一次给一张图片
        image_id = idx
        Input = torch.tensor(np.zeros((400, 400, 3)), dtype = int) # 每个像素点查询信息的构造
        # 下面是比较高效的写法，以下所有填充数据已经在initialize里面计算完成
        Input[:, :, 0] = self.H[image_id]  # 填充第一通道 # modified
        Input[:, :, 1] = self.D[image_id]  # 填充第二通道 # modified
        Input[:, :, 2] = torch.tensor(self.UQueries, dtype = int)  # 使用递增的 count 填充第三通道
        Output = self.get_image(image_id)
        return Input, Output

torch.set_default_tensor_type(torch.cuda.FloatTensor)
if __name__ == '__main__':
# 设置默认tensor都创建到显卡上
    # 数据集实例化
    MyDatasetSample = MyDataset("Leather08.btf")
    # 创建data loader
    dataloaderInstance = DataLoader(dataset = MyDatasetSample,  batch_size = 16, drop_last = False, shuffle = False)

    # 创建模型并移动到cuda上
    model = MLP()
    model.cuda()

    # 根据论文信息设置对应的学习率
    MLP_learning_rate = 3e-3
    FeaturePlane_learning_rate = 1e-2

    # 根据论文信息，使用L1loss作为损失函数
    criterion = torch.nn.L1Loss()
    # 创建adamw优化器，并给不同的层设置不同的学习率
    optimizer = AdamW([
            {'params' : [model.H.weight, model.D.weight, model.U.weight], 'lr' : FeaturePlane_learning_rate},
            {'params' : [model.hidden1.weight, model.hidden2.weight, model.hidden3.weight, model.hidden4.weight], 'lr' : MLP_learning_rate}
        ])


    # 一共50个epoch
    for epoch in range(50):
        # 根据论文信息，每次取16张图片
        count = 0
        print(optimizer.param_groups[0]['lr']) 
        print(optimizer.param_groups[1]['lr']) 
        for data in dataloaderInstance:
           
            FinalInput, target = data
            optimizer.zero_grad()
            yhat = model(FinalInput)
            loss = criterion(yhat, target)
            print("current loss is", loss, "epoch", epoch, "count", count)
            
            count += 1
            loss.backward()
            optimizer.step()

        #  根据论文信息，更新学习率，每次乘以0.9
        # optimizer.param_groups[0]['lr'] *= 0.9
        # optimizer.param_groups[1]['lr'] *= 0.9
        
        # 手动释放缓存
        torch.cuda.empty_cache()

    # 最后保存模型，后续可以直接加载并读取对应层的参数
    torch.save(model, "BTF_MLP_08.pth")
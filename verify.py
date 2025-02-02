import pickle
from BTF2 import MLP
import torch
import os
from btf_extractor import Ubo2014
import numpy as np
import matplotlib.pyplot as plt
from math import log10, sqrt

def PSNR(original, compressed):
    mse = np.mean((original - compressed) ** 2)
    if mse == 0: # MSE为零表示信号中没有噪声
        return 100
    max_pixel = 1
    psnr = 20 * log10(max_pixel / sqrt(mse))
    return psnr

def get_image(id):
    with open("dataset/Leather08/" + str(id) +'.pkl', 'rb') as file:
        color = pickle.load(file)
    return color

with open("dataset/Leather08/" + 'H.pkl', 'rb') as file:
    H = pickle.load(file)
with open("dataset/Leather08/" + 'D.pkl', 'rb') as file:
    D = pickle.load(file)
with open("dataset/Leather08/" + 'U.pkl', 'rb') as file:
    U = pickle.load(file)



model = torch.load('BTF_MLP_08_30.7.pth')
model.eval()
for i in range(2000):
    id = i
    Input = np.zeros((400, 400, 3))
    Input[... , 0] = H[id]
    Input[... , 1] = D[id]
    Input[... , 2] = U
    picture = model.forward(torch.tensor(Input, dtype = int)).cpu().detach().numpy()

    ds_dir = os.path.join("dataset","UBO2014")
    btf_path = os.path.join(ds_dir, "leather08.btf")
    btf = Ubo2014(btf_path) # 第一变量BTF本身

    angles = list(btf.angles_set)
    angles = np.array(angles) # 第二变量所有角度

    picture2 = btf.angles_to_image(*angles[id])
    picture2 = picture2[..., ::-1]

    # 创建一个新的图形
    plt.figure()

    # 添加第一个子图
    plt.subplot(1, 2, 1) # (行数, 列数, 面板编号)
    plt.imshow(picture)
    plt.title('Image 1')
    plt.axis('off') # 关闭坐标轴

    # 添加第二个子图
    plt.subplot(1, 2, 2)
    plt.imshow(picture2)
    plt.title('Image 2')
    plt.axis('off')

    value = PSNR(picture, picture2)
    print(f"PSNR value is {value} dB")

    # 显示图形
    plt.show() 
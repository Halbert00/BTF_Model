import os
from btf_extractor import Ubo2014
import pickle
from math import cos, sin, acos, atan2
import math
import numpy as np

def AngleQuery(theta, phi): 
    # 输入球坐标（theta, phi），输出int，代表该坐标在20*20分辨率下的编码
    # theta ∈ [0, pi / 2]
    # phi 属于[0, 2pi)    
    theta = math.degrees(theta)
    phi = math.degrees(phi)
    phi %= 360
    theta *= 2
    theta = theta // 9
    phi = phi // 18
    assert(theta * 20 + phi < 400)
    return int(theta * 20 + phi)

def cartesian_to_spherical(Vector):
    # 辅助函数1，把3D坐标转球坐标，要求Vector的模是1
    # 输出球坐标，范围[0, pi]和[0, 2pi)
    assert(len(Vector) == 3)
    phi = atan2(Vector[1], Vector[0]) # atan2范围是-Pi 到 Pi，需要矫正
    theta = acos(Vector[2]) # acos的范围是0- pi，不需要矫正
    if phi < 0:
        phi += math.pi * 2 #矫正atan2的结果
    return [theta, phi]
    
def YRotate(OriginalSperical, angle): # 坐标绕Y轴旋转
    thetaO = OriginalSperical[0]
    phiO = OriginalSperical[1]
    theta = acos(max(min(-sin(angle) * sin(thetaO) * cos(phiO) + cos(angle) * cos(thetaO), 1), -1))
    phi = atan2(sin(thetaO) * sin(phiO), cos(angle) * sin(thetaO) * cos(phiO) + sin(angle) * cos(thetaO))
    if phi < 0:
        phi += math.pi * 2
    return [theta, phi]
    
def Rusin(angle):
    assert(len(angle) == 4)
    theta1 = math.radians(angle[0])
    phi1 = math.radians(angle[1])
    theta2 = math.radians(angle[2])
    phi2 = math.radians(angle[3])

    # 恢复两个3D坐标
    omegaIn = [cos(phi1) * sin(theta1), sin(phi1) * sin(theta1), cos(theta1)]
    omegaOut = [cos(phi2) * sin(theta2), sin(phi2) * sin(theta2), cos(theta2)]
        
    # 相加，并归一，得到half-vector H
    addedH = np.array(omegaIn) + np.array(omegaOut)
    addedHModule = math.sqrt(addedH[0] ** 2 + addedH[1] ** 2 + addedH[2] ** 2)
    VectorH = [addedH[0] / addedHModule, addedH[1] / addedHModule, addedH[2] / addedHModule]
    # 从3D坐标恢复球坐标
    spericalH = cartesian_to_spherical(VectorH)

    # 第一次旋转，绕Z轴转H的-phi°，对于球坐标而言，只是phi相减即可
    # 此时对应的H转到了X-Z平面上，角度为坐标为[thetaH, 0]
    omegaInFirstRotate = [theta1, phi1 - spericalH[1]]
    # 第二次旋转，绕Y轴旋转-thetaH，调用之前写好的方法
    # 此时对应的H转到了Z轴重合，完成了所谓的移动到NorthPole
    omegaInFinal = YRotate(omegaInFirstRotate, -spericalH[0])
    # 该函数只需要计算两万次(151*151)，所以优化并不重要，直接在开始的时候计算好即可，后面查表
    return [spericalH[0], spericalH[1], omegaInFinal[0], omegaInFinal[1]]

def get_image(id): # 从BTF中提取图片，似乎性能还好，接近C的理论表现
    image = btf.angles_to_image(*angles[id])
    image = image[...,::-1]
    return image

ds_dir = os.path.join("dataset","UBO2014")
btf_path = os.path.join(ds_dir, "leather08.btf")
btf = Ubo2014(btf_path) # 第一变量BTF本身

angles = list(btf.angles_set)

# 检查目录是否存在
if not os.path.exists("dataset/Leather08"):
    print("创建解压目录")
    os.mkdir("dataset/Leather08")

for i in range(151 * 151):
    with open("dataset/Leather08/"+ str(i) + '.pkl', 'wb') as file:
        pickle.dump(get_image(i), file)

H = np.zeros(len(angles)) # 开始计算所有图片的查询H
D = np.zeros(len(angles)) # 开始计算所有图片的查询D

# 在初始化的时候就完成所有的编码工作
for i in range(len(angles)):
    RPedcoordinate = Rusin(angles[i])
    H[i] = AngleQuery(RPedcoordinate[0], RPedcoordinate[1]) # 在完成计算后引入分辨率，编码成0-400的int
    D[i] = AngleQuery(RPedcoordinate[2], RPedcoordinate[3]) # 同上

UQueries = np.arange(400 * 400).reshape(400, 400) # 所有的U查询int也可以在这里初始化完成

with open("dataset/Leather08/" + 'H.pkl', 'wb') as file:
    pickle.dump(H, file)
with open("dataset/Leather08/" + 'D.pkl', 'wb') as file:
    pickle.dump(D, file)
with open("dataset/Leather08/" + 'U.pkl', 'wb') as file:
    pickle.dump(UQueries, file)

# 以上所有数据可以被复用到每次get item中，并且只读，所以后续只需要考虑图片加载即可
print("initialize finished")
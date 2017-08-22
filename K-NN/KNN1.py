# -*- coding: UTF-8 -*-

import numpy as np
import random
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap


# 
def GaussNum(mean,sigma,count):
    res = [0] * count
    for i in range(count):
        res[i] = random.gauss(mean,sigma) 
    return res

# 计算点与点的距离
def Distance_for_dot(inX,x,y):
    size = len(inX)
    sum = 0
    for i in range(size):
        sum += ((inX[0] - x) ** 2 + (inX[1] - y) ** 2)
    return sum ** 0.5

def knn_Classify(inX,x,y,labels,k):
    dataSetSize = len(x)
    distances_for_eachSample = [0] * dataSetSize
    print(type(distances_for_eachSample))
    data_and_label = {}
    for i in range(dataSetSize):
        distances_for_eachSample[i] = Distance_for_dot(inX,x[i],y[i])
        data_and_label[distances_for_eachSample[i]] = labels[i]
    
    # 这里排序完成后，data_and_labels变为list型
    # 按键值对（key）排序
    data_and_label = sorted(data_and_label.items(),key = lambda d:d[0])
    print((data_and_label))
    res = {}
    for i in range(k):
        if (res.__contains__(data_and_label[i][1])):
            # print(data_and_label[i][1])
            res[data_and_label[i][1]] += 1
        else:
            res[data_and_label[i][1]] = 1
    
    print(res)
    res = sorted(res.items(),key = lambda d:d[1],reverse = True)
    print((res))
    print("the result is : %s" %res[0][0])


x1 = GaussNum(15,7,200)
y1 = GaussNum(20,7,200)
print(type(x1))
x2 = GaussNum(30,7,200)
y2 = GaussNum(50,8,200)

x3 = GaussNum(5,7,200)
y3 = GaussNum(40,7,200)

plt.scatter(x1,y1,c='b',marker='s',s=50,alpha=0.8)
plt.scatter(x2,y2,c='r',marker='^',s=50,alpha=0.8)
plt.scatter(x3,y3,c='g',marker='o',s=50,alpha=0.8)
plt.show()
print(type(x1 + x2 + x3))
x = x1 + x2 + x3
y = y1 + y2 + y3
labels = [1]*200+[2]*200+[3]*200
print(type(labels))
knn_Classify([20,40],x,y,labels,50)




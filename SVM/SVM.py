from svmMLiA import *
from numpy import *
# import numpy as np


def somSimple(dataMatIn, classLable, C, toler, maxIter):
    # 载入样本和标签，并初始化 alpha、iter
    dataMatrix = mat(dataMatIn)
    labelMatrix = mat(classLable).T
    b = 0
    m, n = shape(dataMatrix)
    alphas = mat(zeros((m, 1)))
    iter = 0
    
    while(iter < maxIter):
        alphaPairsChanged = 0
        for i in range(m):
            fxi = float(multiply(alphas, labelMatrix).T * (dataMatrix * dataMatrix[i,:].T)) + b
            Ei = fxi - float(labelMatrix[i])
            if(((labelMatrix[i] * Ei < -toler) and (alphas[i] < C)) or
                ((labelMatrix[i] * Ei > toler) and (alphas[i] > 0))):
                j = selectJrand(i, m)
                fxj = float(multiply(alphas, labelMatrix).T * (dataMatrix * dataMatrix[j, :].T)) + b
                Ej = fxj - float(labelMatrix[j])
                alphaIold = alphas[i].copy()
                alphaJold = alphas[j].copy()
                if(labelMatrix[i] != labelMatrix[j]):
                    L = max(0, alphas[j] - alphas[i])
                    H = min(C, C + alphas[j] -alphas[i])
                else:
                    L = max(0, alphas[j] + alphas[i] - C)
                    H = min(C, alphas[j] + alphas[i])
                if(L == H):
                    print("L == H !")
                    continue
                eta = 2.0 * dataMatrix[i,:] * dataMatrix[j,:].T - dataMatrix[i,:] * dataMatrix[i,:].T - dataMatrix[j,:] * dataMatrix[j,:].T
                if(eta >= 0):
                    print("eta >= 0 !")
                    continue
                alphas[j] -= labelMatrix[j] * (Ei - Ej) / eta
                alphas[j] = clipAlpha(alphas[j],H,L)
                if((abs(alphas[j] - alphaJold) < 0.00001)):
                    print(" j not moving enough ")
                    continue
                alphas[i] += labelMatrix[j]* labelMatrix[i] * (alphaJold - alphas[j])
                b1 = b - Ei - labelMatrix[i] * (alphas[i] - alphaIold) * dataMatrix[i,:] * dataMatrix[i,:].T
                - labelMatrix[j] * (alphas[j] - alphaJold) * dataMatrix[i,:] * dataMatrix[j,:].T
                b2 = b - Ej - labelMatrix[i] * (alphas[i] - alphaIold) * dataMatrix[i, :] * dataMatrix[j, :].T
                - labelMatrix[j] * (alphas[j] - alphaJold) * dataMatrix[j, :] * dataMatrix[j, :].T
                
                if((0 < alphas[i]) and (C > alphas[i])):
                    b = b1
                elif((0 < alphas[j]) and (C > alphas[j])):
                    b = b2
                else:
                    b = (b1+b2) / 2.0
                alphaPairsChanged += 1
                print("iter: %d i: %d, pairs changed %d" %(iter, i, alphaPairsChanged))
        if(alphaPairsChanged == 0):
            iter += 1
        else:
            iter = 0
        print(" ************************** iteration number : %d" %iter)
    return b,alphas
    
    

if __name__ == '__main__':
    # 载入数据
    data, label = loadDataSet("testSet.txt")
    # 绘制画布，将后面的点、线都画在同一个画布
    fig = plt.figure(figsize=(8, 6))
    f1 = fig.add_subplot(1, 1, 1)
    # 使用简单som计算svm分类平面需要的参数
    b, alphas = somSimple(data, label, 0.6, 0.0001, 50)
    ws = calcWs(alphas, data, label)
    # 打印参数
    print("b = " + str(b))
    print("\n")
    print("alphas = "+ str(alphas))
    
    # 画出样本集
    plotDataSet(data, label, f1)
    # 画出使用简单som找到的支持向量
    plotSVCircle(data, alphas, f1)
    # 画出本次支持向量分类平面
    plotSVMDecision(ws, b, f1)
    plt.show()
    

    
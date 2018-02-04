from numpy import *
import matplotlib.pyplot as plt
import matplotlib.patches

# 打开并读入数据样本
def loadDataSet(fileName):
    dataMat = []
    labelMat = []
    fileread = open(fileName)
    for line in fileread.readlines():
        lineArr = line.strip().split('\t')
        dataMat.append([float(lineArr[0]), float(lineArr[1])])
        labelMat.append(float(lineArr[2]))
    return dataMat, labelMat

def selectJrand(i,m):
    j = i
    while (j == i):
        j = int(random.uniform(0,m))    # uniform(i,j,k) 在[i,j)之间产生k个随机数（float）
    return j

def clipAlpha(aj,H,L):
    if aj > H:
        aj = H
    if L > aj:
        aj = L
    return aj

# 画数据集
def plotDataSet(dataMat, labelMat, figure):
    dataMatrix = mat(dataMat)
    labelMatrix = mat(labelMat).T
    print(dataMatrix.shape)
    # 将标签不同的两类训练样本分别画出
    xcord1 = [];    ycord1 = []
    xcord0 = [];    ycord0 = []
    # 将训练样本按照标记不同，分为两类不同的点
    for i in range(dataMatrix.shape[0]):
        if int(labelMatrix[i]) == -1:
            xcord0.append(dataMatrix[i,0])
            ycord0.append(dataMatrix[i,1])
        else:
            xcord1.append(dataMatrix[i,0])
            ycord1.append(dataMatrix[i,1])

    figure.scatter(xcord0, ycord0, s = 30,  marker = 's',c = 'red')
    figure.scatter(xcord1, ycord1, s = 30, marker = 'o', c = 'blue')
    plt.title("SVM")

    


# 画出支持向量
def plotSVCircle(dataMat,alphas, figure):
    dataMatrix = mat(dataMat)
    # 记录支持向量的下标
    # sv = []
    for i in range(dataMatrix.shape[0]):
        if alphas[i] > 0.0:
            circle = plt.Circle((dataMatrix[i, 0], dataMatrix[i, 1]), 0.5, facecolor='none', edgecolor=(0, 0.8, 0.8),
                                linewidth=3, alpha=0.5)
            figure.add_patch(circle)


# 求出分类平面所需要的参数W
def calcWs(alphas, dataArr, classLabels):
    """
    根据支持向量计算分离超平面(w,b)的w参数
    :param alphas:拉格朗日乘子向量
    :param dataArr:数据集x
    :param classLabels:数据集y
    :return: w=∑alphas_i*y_i*x_i
    """
    X = mat(dataArr);
    labelMat = mat(classLabels).transpose()
    m, n = shape(X)
    w = zeros((n, 1))
    for i in range(m):
        w += multiply(alphas[i] * labelMat[i], X[i, :].T)
    return w

# 画出分类平面
def plotSVMDecision(w, b, figure):
    w0 = w[0][0]
    w1 = w[1][0]
    b = float(b)
    x = arange(-2.0, 12.0, 0.1)
    y = (-w0 * x - b) / w1
    figure.plot(x, y)
    figure.axis([-2, 12, -8, 6])
    
            

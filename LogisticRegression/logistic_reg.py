#
'''
	1	#这里split(‘str a’)函数是将读入的一行数据用a为分割符分割开
		#如果提示：could not convert string to float，则说明读入的数据中有些
		#字符串并不能转化为float类型，检查数据之间是否有多个空格，或者Tab符，
		#常用的分隔符：'\t','  '
	2	# numpy.matrix.getA()：
		# Return `self` as an `ndarray` object.
	'''

from numpy import *
import matplotlib.pyplot as plt

print('\n###############logistic regression#####################')

# 载入数据
def loadDataSet():

    dataMat = [];   labelMat = []
    fr = open('testSet.txt')
	# 逐行读入数据，然后strip去头去尾，用split分组
    for line in fr.readlines():
        lineArr = line.strip().split('   ')
        dataMat.append([1.0,float(lineArr[0]),float(lineArr[1])])
        labelMat.append(int(lineArr[2]))
    return dataMat,labelMat

def sigmoid(inX):
    return 1.0/(1+exp(-inX))

# 梯度下降法
def gardDescent(dataMatIn,classLabels):
    dataMatrix = mat(dataMatIn)		
    labelMatrix = mat(classLabels).T
    m,n = shape(dataMatrix)		# 得到数据规模
	# 迭代步长
    alpha = 0.01
    # 迭代次数
    maxCycles = 50000
    weights = ones((n,1))		# help(numpy.ones)
    							# 设定初始参数，全为1
    for k in range(maxCycles):
        h = sigmoid(dataMatrix * weights)
        E = (h - labelMatrix)
        weights = weights - alpha * dataMatrix.T * E
    return weights

# 随机梯度上升算法
# 每次对参数的更新都只用一个样本值，属于在线算法
def stocGradAscent0(dataMatrix,classLabels):
    dataMatrix = array(dataMatrix)
    m,n = shape(dataMatrix)
    alpha = 0.01
    weights = ones(n)
    for j in range(500):
        for i in range(m):
            h = sigmoid(sum(dataMatrix[i] * weights))
            E = classLabels[i] - h
			weights = weights + alpha * E * dataMatrix[i]

	return weights


# 画图函数，传入参数为两种算法得到的参数矩阵
def plotBestFit(weights_1,weights_2):
	# numpy.matrix.getA()：
	# Return `self` as an `ndarray` object.
	weights_1 = weights_1.getA()
	# weights_2 = weights_2.getA()
	dataMat,labelMatrix = loadDataSet()
	dataArr = array(dataMat)
	n = shape(dataArr)[0]
	# 将标签不同的两类训练样本分别画出
	xcord1 = []; ycord1 = []
	xcord2 = []; ycord2 = []
	# 将训练样本按照标记不同，分为两类不同的点
	for i in range(n):
		if int(labelMat[i]) == 1:
			xcord1.append(dataArr[i,1])
			ycord1.append(dataArr[i,2])
		else:
			xcord2.append(dataArr[i,1])
			ycord2.append(dataArr[i,2])

	fig = plt.figure(figsize=(14,6))

	# 图1
	ax = fig.add_subplot(121)
	ax.scatter(xcord1,ycord1,s=30,c='red',marker='s')
	ax.scatter(xcord2,ycord2,s=30,c='blue',)
	x = arange(-3.0,3.0,0.1)
	y_1 = (-weights_1[0]-weights_1[1]*x)/(weights_1[2])
	ax.plot(x,y_1,'k--',color = 'yellow',linewidth=2)
	# plt.xlabel('x1');plt.ylabel('x2');
	plt.xlabel('Logistics Regression GradDescent')
	ax.spines['right'].set_color('none')
	ax.spines['top'].set_color('none')
	
	# 图2
	ax = fig.add_subplot(122)
	ax.scatter(xcord1,ycord1,s=30,c='red',marker='s')
	ax.scatter(xcord2,ycord2,s=30,c='blue',)
	x = arange(-3.0,3.0,0.1)
	y_2 = (-weights_2[0]-weights_2[1]*x)/(weights_2[2])
	ax.plot(x,y_2,'k--',color = 'yellow',linewidth=2)
	# plt.xlabel('x1');plt.ylabel('x2');
	plt.xlabel('Logistics Regression StocGradDescent')
	# 将坐标系右边和上边去掉，美观
	ax.spines['right'].set_color('none')
	ax.spines['top'].set_color('none')
	

	plt.show()


# 准确度计算
def calAccuracyRate(dataMat,labelMat,weights):
	count = 0
	dataMat = mat(dataMat)
	labelMat = mat(labelMat).T
	m,n = shape(dataMat)

	for i in range(m):
		h = sigmoid(dataMat[i,:] * weights)
		if ( h>0.5 and int(labelMat[i,0]) == 1) or ( h<0.5 and int(labelMat[i,0]) == 0 ):
			count += 1 
	#	elif ( h<0.5 and int(labelMat[i,0]) == 0 ):
	#		count += 1
	return count/m

dataMat,labelMat = loadDataSet()
weights_GD = gardDescent(dataMat,labelMat)				# 使用梯度下降计算参数矩阵 θ
weights_SGD = stocGradAscent0(dataMat,labelMat)			# 使用随机梯度下降计算参数矩阵 θ

print('weights_GD:\n',weights_GD)
print('weights_SGD:\n',weights_SGD)
plotBestFit(weights_GD,weights_SGD)

acc_gd = calAccuracyRate(dataMat,labelMat,weights_GD)
#print(type(weights_SGD))
weights_SGD = mat(weights_SGD).transpose() 
#print(type(weights_SGD))
#print(weights_SGD)

acc_sgd = calAccuracyRate(dataMat,labelMat,weights_SGD)

print('\n\nacc_gd:',acc_gd) 
print('acc_sgd:',acc_sgd) 
print('\n\n')


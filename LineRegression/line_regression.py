# coding=UTF-8

# 最小二乘法的线性回归，也是基于一维特征性向量的线性回归

from numpy import *
import matplotlib.pyplot as plt

''' 从本地 .txt 文件载入数据，将特征向量赋值给dataMat，
	输出标记赋值给laberMat，并return结果

	Str.split(scbep=None, maxsplit=-1) -> list of strings
    	字符串分割，sep为分隔符，默认值为空格

	
	Str.strip([chars]) -> str
    	Return a copy of the string S with leading 
    and trailing whitespace removed.
    	If chars is given and not None, remove 
    characters in chars instead.

'''
def loadDataSet(fileName):
	# 记录数据的列数 =2
	numFeat = len(open(fileName).readline().split('\t')) - 1
	
	dataMat = [];	labelMat = []		# 将数据集分为 自变量 和 标签量
	fr = open(fileName)			# 打开文件
	
	for line in fr.readlines():			# 按行读入文件中的数据
		lineArr = []
		curLine = line.strip().split('\t')		# 将当前行去掉前后空格，按照‘/t’分割，存入curline数组
		for i in range(numFeat):		# 将自变量数据依次读入数据 前n-1列
			lineArr.append(float(curLine[i]))
		dataMat.append(lineArr)		# 此行数据加入自变量数据列表
		labelMat.append(float(curLine[-1]))		# curLine[-1] 表示当前行数据的最后一列，即数据标签值。这里等价于curLine[2]
	
	return dataMat,labelMat


''' 线性回归函数，两个参数 xArr：特征向量  yArr：标签向量
'''
def standRegeres(xArr,yArr):
	xMat = mat(xArr)		#xMat  2*n矩阵
	yMat = mat(yArr).T  	#yMta  n*1矩阵
	xTx = xMat.T * xMat		#xTx   2*2矩阵
	if linalg.det(xTx) == 0.0:
		print("This matrix is singular,cannot do inverse")
		return
#  最小二乘法计算回归系数ws
	ws = xTx.I * (xMat.T * yMat)	#ws (2*2) * ( (2*n)*(n*1) )矩阵
	return ws



# 调用上述两个函数，载入数据处理并计算回归系数
xArr,yArr = loadDataSet('ex1.txt')
ws = standRegeres(xArr,yArr)

# 为画图计算
xMat = mat(xArr);	yMat = mat(yArr)
yHat = xMat*ws


# 画图
fig = plt.figure()
ax = fig.add_subplot(111)
# 画数据点，即散点图
ax.scatter(xMat[:,1].flatten().A[0],yMat.T[:,0].flatten().A[0])
# 画线
xCopy = xMat.copy()	# 得到XMat一个复制矩阵
xCopy.sort(0)		# 矩阵排序，保证画图时顺序正确
yHat = xCopy*ws
print(type(yHat))
ax.plot(xCopy[:,1],yHat)

plt.title('Linear Regression')
plt.xlabel('X')
plt.ylabel('Y')	
plt.grid(True)		# 显示网格

plt.show()

# 计算相关系数，这是需要将yMat转置，保证两个向量都是行向量
print(corrcoef(yHat.T,yMat))
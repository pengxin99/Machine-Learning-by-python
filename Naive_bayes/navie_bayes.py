# navie bayes
import numpy as np
'''
# 载入文档及文档类别
def loadDataSet():

	postingList = [['my','dog','has','flea','problems','help','please'],
					['maybe','not','take','him','to','dog','park','stupid'],
					['my','dalmation','is','so','cute','i','love','him'],
					['stop','posting','stupid','worthless','garbage'],
					['mr','licks','ate','my','steak','how','to','stop','him'],
					['quit','buying','worthless','dog','food','stupid']]
	classVec = [0,1,0,1,0,1]
	return postingList,classVec
'''

# 载入数据
def loadDataSet():
    dataMat = [];   labelMat = []
    fr = open('testdata.txt')
    for line in fr.readlines():		
    	temp = []
    	lineArr = line.strip().split(' ')		# 逐行读入数据，然后strip去头去尾，用split分组
    	for i in range(len(lineArr) - 2):		# 将数据按行读入，最后一位为标签位，其他为需要判读的语句
    		temp.append(lineArr[i])
    	
    	dataMat.append(temp)					# 读入每行的语句
    	labelMat.append( int(lineArr[-1]) )		# 读入每行语句的标签

    return dataMat,labelMat



def creatVocabList(dataSet):		# 创建词向量，即将出现过得词都放在一个集合中（无重复）
	vocabSet = set([])
	for document in dataSet:
		vocabSet = vocabSet | set(document)
	return list(vocabSet)


def setOfWords2Vec(testList,myVocabVec):		# 将文档转化为向量
	Vec = [0] * len(myVocabVec)
	for word in testList:
		if word in myVocabVec:
			Vec[myVocabVec.index(word)] = 1
		else:
			print('the word: %s is not in my Vocabulary!' %word)
	return Vec


def createTrainMatrix(trainListing,myVocabVec):		# 生成训练矩阵，即每个样本的特征向量
	trainMatrix=[]   #训练矩阵
	for i in range(len(trainListing)):
		curVec = setOfWords2Vec(trainListing[i],myVocabVec)
		trainMatrix.append(curVec)
	return trainMatrix
	
def trainNB0(trainMatrix,tarinCategory):		# 
	numTrainDocs = len(trainMatrix)		# 文档数量
	numWords = len(trainMatrix[0])		# 样本特征数，这里等于构建的词向量的长度 ==len(myVocabVec)
	
	pAbusive = sum(tarinCategory)/float(numTrainDocs)		# 类别为1的文档数的占比，即p(1)
	p0Num = np.ones(numWords); p1Num = np.ones(numWords)		# 对于不同类别，建立单词统计矩阵，可以按位得到每个单词的数量
	p0Denom = 2.0; p1Denom = 2.0		# 对于不同类别，统计总单词数
	
	for i in range(numTrainDocs):
		if tarinCategory[i] == 1:
			p1Num += trainMatrix[i]		# 对于类别1，按位得到每个单词的数量
			p1Denom += sum(trainMatrix[i])		# 对于类别1，统计总单词数
		else:
			p0Num += trainMatrix[i]
			p0Denom += sum(trainMatrix[i])

	p1Vect = p1Num/p1Denom				# 向量除法，对应位相除，得到每个词在类别1下的概率，即 p(w0|c=1),p(w0|c=1).....
	p0Vect = p0Num/p0Denom									
	
	p1Vect = np.log(p1Vect) 					# 取对数，之后的乘法就可以改为加法，防止数值下溢损失精度
	p0Vect = np.log(p0Vect)
	return p0Vect,p1Vect,pAbusive		

# 朴素贝叶斯分类，得到测试用例为0还是1
def NBclassify(testDoc,myVocabVec,p0Vect,p1Vect,pClass1):
	vec2Classify = setOfWords2Vec(testDoc,myVocabVec)
	p1 = sum(vec2Classify * p1Vect) + np.log(pClass1)
	p0 = sum(vec2Classify * p0Vect) + np.log(1 - pClass1)
	if p1 > p0:
		return 1
	else:
		return 0



##########################
listing,listclass = loadDataSet()
myVocabVec = creatVocabList(listing)
#print(myVocabVec)
print(listing,listclass)
#v0 = setOfWords2Vec(listing[0],myVocabVec)
#print(v0)
trainMatrix = createTrainMatrix(listing,myVocabVec)

#for i in range(len(trainMatrix)):
#	print(trainMatrix[i])

p0Vect,p1Vect,pClass1 = trainNB0(trainMatrix,listclass)


testEntry0 = ['i','love','you']
testEntry1 = ['stupid','you']

result0 = NBclassify(testEntry0,myVocabVec,p0Vect,p1Vect,pClass1)
result1 = NBclassify(testEntry1,myVocabVec,p0Vect,p1Vect,pClass1)
print(result0,result1)
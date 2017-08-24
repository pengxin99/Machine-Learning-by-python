# 从网博客参考而来

from numpy import *  
import operator  
  
class KNN:  
    def createDataset(self):  
        group = array([[1.0,1.1],[0,0],[1.0,1.0],[0,0.1]])  
        labels = ['B','A','B','A']  
        return group,labels  
  
    def KnnClassify(self,testX,trainX,labels,K):  
        [N,M]=trainX.shape  
      
    # calculate the distance between testX and other training samples  
    # tile for array and repeat for matrix in Python, == repmat in Matlab  
    # 对testX矩阵做 (N,1)的重复，得到与trainX同样的维度，这里的N为测试数据样本的个数，
    # 完成tile(testX,(N,1))之后方便与每个测试样本做运算
        
        difference = tile(testX,(N,1)) - trainX 
        print(difference)
        difference = difference ** 2 # take pow(difference,2)  
        print(difference)
        # sum(axis=1)按行求和，即没行的元素相加，得到
        distance = difference.sum(1) # take the sum of difference from all dimensions  
        print(distance)
        distance = distance ** 0.5  
        print(type(distance))
        ############# 此处 argsort() 用的精华，返回的是索引，而不是原数组
        sortdiffidx = distance.argsort()  
        print(sortdiffidx)
      
    # find the k nearest neighbours  
        vote = {} #create the dictionary  
        for i in range(K):  
            ith_label = labels[sortdiffidx[i]];  
            vote[ith_label] = vote.get(ith_label,0)+1 #get(ith_label,0) : if dictionary 'vote' exist key 'ith_label', return vote[ith_label]; else return 0  
        sortedvote = sorted(vote.items(),key = lambda x:x[1], reverse = True)  
        # 'key = lambda x: x[1]' can be substituted by operator.itemgetter(1)  
        return sortedvote[0][0]  
  
k = KNN() #create KNN object  
group,labels = k.createDataset()  
cls = k.KnnClassify([0,0],group,labels,3)  
print(cls)
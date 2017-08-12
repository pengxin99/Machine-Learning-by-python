##Study.163.com Numpy学习

import numpy as np 


'''mat 与 arr 的区别：
matrix是array的分支，matrix和array在很多时候都是通用的，
你用哪一个都一样。但这时候，官方建议大家如果两个可以通用，
那就选择array，因为array更灵活，速度更快，很多人把二维的array
也翻译成矩阵。
但是matrix的优势就是相对简单的运算符号，比如两个矩阵相乘，就是用符号*，
但是array相乘不能这么用，得用方法.dot()
array的优势就是不仅仅表示二维，还能表示3、4、5...维
'''

array = np.array([[1,2,3,7,8,9]])
print(array)
#矩阵维数 ndim
print('number of dim:',array.ndim)
#矩阵大小 shape，返回矩阵行数和列数
print('shape:',array.shape)
#矩阵元素个数 size
print('size:',array.size)


#####课时5：Numpy基础运算1#################################

#arange(start,stop,step) 生成从start到stop，步长为step的矩阵
a = np.arange(4)
#reshap(a,b)  将矩阵重新变换为 a*b 矩阵
b = np.arange(4).reshape(2,2)
c = ([[0,1],
	  [2,3]])
print(a)
print(b)
#求两个矩阵相乘的两种方法，但是要注意两矩阵维数必须符合乘法规则
print(np.dot(b,c))
print(b.dot(c))

#随机生成矩阵，元素为 0-1之间，大小为 a*b
a = np.random.random((2,4))
print(a)
#相应的和、最大、最小值，
#axis=1为求该行，axis=0为求该列
print(np.sum(a,axis=1))
print(np.min(a,axis=0))
print(np.max(a,axis=1))


#####课程6：Numpy基础运算2############################

A = np.arange(2,14).reshape((3,4))
#求矩阵最大、最小元素的索引
print(np.argmin(A))
print(np.argmax(A))
#求矩阵平均值
print(np.mean(A))
print(A.mean())
print(np.average(A))
#求矩阵中位数
print(np.median(A))
#累加，地n位元素为前n-1个元素之和
print(A)
print(np.cumsum(A))
#累加，地n位元素为前n-1个元素之和
print(np.diff(A))
#矩阵转置
print(np.transpose(A))
print((A.T))
#输出非零元素的坐标值
print(np.nonzero(A))

#截取功能 
#	clip(a,a_min,a_max,out=None)
#	比a_min小的元素都置为a_min,比a_max大的元素都置为a_max
print(np.clip(A,5,9))
#排序,逐行排序
A = np.random.random((2,4))
print(A)
print(np.sort(A))


#########课程7 Numpy索引 ##################

A = np.arange(3,15).reshape((3,4))

#A = ([['zhao','qian','sun','li'],
#	 ['zhou','wu','zhen','wang'],
#	 ['feng','chen','chu','wei']])
#A.reshape((3,4))

print(A)
print(A[2][1])
print(A[2,1])
#打印第二行所有元素
print(A[2,:])
#打印第一列所有元素
print(A[:,0])
#迭代输出每一行，默认为行，要想输出列，可将矩阵装置
for row in A:
	print(row)
#迭代输出每一列,
for column in A.T:
	print(column)
#迭代输出每一列，先将矩阵flat，变成一行元素
for item in A.flat:
	print(item)


#########课程8 Numpy的array合并 ##################

A = np.array([1,1,1])
B = np.array([2,2,2])
#vstack()	Stack arrays in sequence vertically
print(np.vstack((A,B,B)))
#horizontal stack
print(np.hstack((A,B,)))
#################!!!将行向量变为列向量，在制定位置添加新坐标！
C = np.array([1,1,1])[:,np.newaxis]
D = np.array([2,2,2])[:,np.newaxis]
#print(C);print(D)
#concatenate()  什么鬼？？
E = np.concatenate((A,B), axis = 0)
print(E)


#########课程9 Numpy的array分割 ##################
A = np.arange(12).reshape(3,4)
print(A)
#split(ary,indices_or_sections,axis=)  把矩阵分成相等的份数
print(np.split(A,3,axis = 0))
#不等量分割,然而每一项是多少，怎么定？？？
print(np.array_split(A,3,axis = 1))
#横项分割，列项分割
print(np.vsplit(A,3))
print(np.hsplit(A,2))


#########课程10 Numpy的copy&deep copy ##################
#python 语言的赋值为引用赋值，本体改变，其他赋值引用所得到的值均改变
#如果想要正常的复制，则需要用到copy(),即为 deep copy。
a = array(10)
print(a)
b = a.copy()		#deep copy
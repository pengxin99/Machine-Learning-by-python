
from PIL import Image
import numpy as np
import math
import matplotlib.pyplot as plt
from numpy import linalg as la


# 图片转换为矩阵
def ImageToMatrix(filename):
    # 读取图片
    im = Image.open(filename)
    # 显示图片
    width,height = im.size
    im = im.convert("L") 
    data = im.getdata()
    data = np.matrix(data,dtype='float')/255.0
    #new_data = np.reshape(data,(width,height))
    new_data = np.reshape(data,(height,width))
    return new_data


# 举证转换为图片
def MatrixToImage(data):
    data = data*255
    new_im = Image.fromarray(data.astype(np.uint8))
    return new_im



# SVD 分解函数

def ImageSVD_bylib(imageData,r = 100):
	## imageDate 输入数据
	## r 该SVD的sigma矩阵取前r个特征值
	## return SVD后的矩阵
	m,n = imageData.shape
	print(m,n)
	min_shape = min(m,n)
	U,sigma,VT=la.svd(imageData)
	

	U_r = U[:,0:r]
	VT_r = VT[0:r,:]
	sigma = sigma[0:r]
	
	S = np.zeros((r, r), dtype=complex)
	S[:r, :r] = np.diag(sigma)
	print(S)

	data_SVD = np.dot(U_r, np.dot(S, VT_r))
	print(np.allclose(imageData, np.dot(U_r, np.dot(S, VT_r))))
	return data_SVD

'''
# SVD 分解函数
def ImageSVD(imageData,k = 10):
	m, n = imageData.shape
	k = n
	imageData_T = imageData.T
	# print(np.dot(imageData,imageData_T))
	# print(np.dot(imageData_T,imageData))

	print("the shape of image is :" + str(imageData.shape))
	# 特征值赋值给a，特征向量赋值给b
	U_a, U_b = np.linalg.eig(np.dot(imageData, imageData_T))
	V_a, V_b = np.linalg.eig(np.dot(imageData_T, imageData))

	print("U_b = " + str(U_b))
	print("V_b = " + str(V_b))

	sorted_indices = np.argsort(V_a)
	topk_Vb = V_b[:, sorted_indices[:-k - 1:-1]] * -1

	U_a = np.matrix(U_a)
	U_b = np.matrix(U_b)
	V_a = np.matrix(V_a)
	V_b = np.matrix(topk_Vb)


	print("the length of U_a is :" + str(U_a.shape))
	print("the shape of U_b is :" + str(U_b.shape))
	print("the length of V_a is :" + str(V_a.shape))
	print("the shape of V_b is :" + str(V_b.shape))
	
	U_a = U_a.reshape(m, 1)
	U = U_b.reshape(m, m)
	V_a = V_a.reshape(n, 1)
	V = V_b.reshape(n, n)
	print("U = " + str(U))

	print("V_sort = " + str(V))
	S = np.zeros((m, n))
	for i in range(min(m, n)):
		S[i][i] = math.sqrt(U_a[i][0])
		# print("****" + str(S[i][i]))

	# print("the shape of U is :" + str(U.shape))
	# print("the shape of v is :" + str(V.shape))
	# print("the length of S is :" + str(S.shape))
	temp = np.dot(U, S)
	# print(V[2])

	print("V=" + str(V))
	AAA = np.dot(temp, V.T)
	# print(type(AAA[0][0]))
	# print(AAA[0][0].shape)
	print("res = " + str(AAA))
	S_size = np.zeros((k, k))

	return AAA

'''

if __name__ == "__main__":

	# 图片路径
	filename = "/home/eason/Desktop/机器学习/SVD/my_test_images/cat1.jpg"

	# 与图片数据转化为矩阵，并显示原图
	data = ImageToMatrix(filename)
	old_img = MatrixToImage(data)
	old_img.show()
	plt.imshow(old_img)

	# 将图片矩阵数据进行SVD，将结果返回，并绘制SVD后的图片
	new_svd_data = ImageSVD_bylib(data,r = 20)
	new_im = MatrixToImage(new_svd_data)

	# 显示变换后的图片并保存
	new_im.show()
	new_im.save('cat_1.jpg')
	# plt.figure("dog")
	# plt.imshow(new_im)
	# plt.show()
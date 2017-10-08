
from PIL import Image
import numpy as np
import math
# import scipy
import matplotlib.pyplot as plt

from numpy import linalg as la


# SVD 分解函数
def ImageSVD_bylib(imageData,r = 2):

	m,n = imageData.shape
	print(m,n)
	min_shape = min(m,n)
	U,sigma,VT=la.svd(imageData)
	print(U)
	print(sigma)
	print(VT)

	U_r = U[:,0:r]
	VT_r = VT[0:r,:]
	sigma = sigma[0:r]
	print("*****************")
	print(U_r)
	print(sigma)
	print(VT_r)
	S = np.zeros((r, r), dtype=complex)
	S[:r, :r] = np.diag(sigma)
	print(S)

	data_SVD = np.dot(U_r, np.dot(S, VT_r))
	print(np.allclose(imageData, np.dot(U_r, np.dot(S, VT_r))))
	return data_SVD


if __name__ == "__main__":
	A = np.array([[3,1,1],[-1,3,1],[5,4,6],[5,5,5]])
	res = ImageSVD_bylib(A)
	print(res)

#!/usr/bin/env python3
import numpy as np
# import cv2

# Define the first set of 3D points
points_set1 = np.array([[1, 2, 3],
                        [4, 5, 6],
                        [7, 8, 9],
                        [3, 3, 6],
                        [5, 4, 5],
                        [2, 2, 3],
                        [3, 6, 7],
                        [1, 0, 6]])

# Define the second set of 3D points
points_set2 = np.array([[1.5, 1.5, 8.5],
                        [4.4, 3.4, 5.4],
                        [7.7, 2.7, 3.7],
                        [3.9, 4.9, 7.8],
                        [5.9, 9.8, 9.9],
                        [2.4, 7.4, 2.4],
                        [3.1, 2.2, 5.1],
                        [1.4, 6.4, 8.4]])


# Perform element-wise multiplication
result = points_set1.reshape(-1, 1, 3) * points_set2.reshape(-1, 3, 1)

# Reshape the result to get a matrix of size N*9
A = result.reshape(-1, 9)
U, S, Vt = np.linalg.svd(A)
preF = np.transpose(Vt)[:, -1].reshape(3, 3)
U, S, Vt = np.linalg.svd(preF)
S[2] = 0
postF = np.matmul(np.matmul(U, np.diag(S)), Vt)
print("postF:", postF)


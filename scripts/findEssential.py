#!/usr/bin/env python3
import numpy as np

# Define the first set of 3D points
points_set1 = np.array([[1, 2, 1],
                        [3, 4, 1],
                        [3.5, 4.5, 1],
                        [4, 5, 1],
                        [6, 7, 1],
                        [8, 9, 1],
                        [-1, -2, 1],
                        [-3, -4, 1],
                        [-5, -6, 1],
                        [-7, -8, 1],
                        [7, 8, 1]])

# Define the second set of 3D points
points_set2 = np.array([[11, 12, 1],
                        [13, 14, 1],
                        [13.5, 14.5, 1],
                        [14, 15, 1],
                        [16, 17, 1],
                        [18, 19, 1],
                        [9, 8, 1],
                        [7, 6, 1],
                        [5, 4, 1],
                        [3, 2, 1],
                        [17, 18, 1]])


# Flatten the points set into 1d array

def findFundamentalMatrix(points_set1, points_set2):
    result = np.kron(points_set1, points_set2)
    A = result.reshape(-1, 9)
    U, S, Vt = np.linalg.svd(A)
    F = np.transpose(Vt)[:, -1].reshape(3, 3)
    U, S, Vt = np.linalg.svd(F)
    S[2] = 0
    F_rank2 = np.matmul(np.matmul(U, np.diag(S)), Vt)
    print(F_rank2)

def findEssentialMatrix(points_set1, points_set2):
    result = np.kron(points_set1, points_set2)
    A = result.reshape(-1, 9)
    U, S, Vt = np.linalg.svd(A)
    E = np.transpose(Vt)[:, -1].reshape(3, 3)
    U, S, Vt = np.linalg.svd(E)
    I = np.identity(3)
    I[2][2] = 0
    E_rank2 = np.matmul(np.matmul(U, I), Vt)
    print(E_rank2)


#findFundamentalMatrix(points_set1, points_set2)
findEssentialMatrix(points_set1, points_set2)

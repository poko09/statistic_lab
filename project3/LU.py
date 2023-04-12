from project3.matrix_invert import inverse_matrix, det
from project3.matrix_operations import matrix_multiplication, subtract, add


def det_LU(L, U):
    det = 1
    for i in range(len(L)):
        det = det * L[i][i] * U[i][i]
    return det


def LU(A, l):
    n = len(A)
    if n == 1:
        return (np.array([[1]]), A)

    if det(A) == 0:
        return "Macierz nie ma macierzy odwrotnej"

    mid = n // 2

    A11 = [row[:mid] for row in A[:mid]]
    A12 = [row[mid:] for row in A[:mid]]
    A21 = [row[:mid] for row in A[mid:]]
    A22 = [row[mid:] for row in A[mid:]]

    L11, U11 = LU(A11, l)
    U11_inv = inverse_matrix(U11, l)
    L21 = matrix_multiplication(A21, U11_inv, l)
    L11_inv = inverse_matrix(L11, l)
    U12 = matrix_multiplication(L11_inv, A12, l)
    S = subtract(A22,
                 matrix_multiplication(matrix_multiplication(A21, U11_inv, l), matrix_multiplication(L11_inv, A12, l),
                                       l))
    L22 = S
    Ls, Us = LU(S, l)
    U22 = Us
    L22 = Ls

    L = [[0 for _ in range(n)] for _ in range(n)]
    for i in range(n // 2):
        for j in range(n // 2):
            L[i][j] = L11[i][j]
            L[i][j + n // 2] = 0
            L[i + n // 2][j] = L21[i][j]
            L[i + n // 2][j + n // 2] = L22[i][j]

    U = [[0 for _ in range(n)] for _ in range(n)]
    for i in range(n // 2):
        for j in range(n // 2):
            U[i][j] = U11[i][j]
            U[i][j + n // 2] = U12[i][j]
            U[i + n // 2][j] = 0
            U[i + n // 2][j + n // 2] = U22[i][j]

    return (L, U)


import pprint
import scipy
import scipy.linalg  # SciPy Linear Algebra Library
import numpy as np

print(np.array(1))

A = np.array([[1, 2, 7, 2], [1, 1, 3, 3], [4, 0, 1, 0], [4, 5, 6, 4]])
P, L, U = scipy.linalg.lu(A)

A2 = [[24]]
P2, L2, U2 = scipy.linalg.lu(A2)

print(L2)
print(U2)

print("A:")
pprint.pprint(A)

print("P:")
pprint.pprint(P)

print("L:")
pprint.pprint(L)

print("U:")
pprint.pprint(U)

pprint.pprint(matrix_multiplication(L, U, 2))

L_m, U_m = LU(A, 2)

pprint.pprint(np.array(L_m))
pprint.pprint(np.array(U_m))
pprint.pprint(matrix_multiplication(L_m, U_m, 2))

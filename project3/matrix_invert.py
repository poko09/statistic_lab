import numpy as np

from project2.mat_mul_op_count import matrix_multiplication_operation_count
from project2.matrix_operations import subtract, matrix_multiplication, add


def identity(n):
    return np.identity(n)


def det(A):
    return np.linalg.det(A)


def inverse_matrix(A, l):
    n = len(A)

    if n == 1:
        return 1 / np.array(A)

    if det(A) == 0:
        return "Macierz nie ma macierzy odwrotnej"

    mid = n // 2

    A11 = [row[:mid] for row in A[:mid]]
    A12 = [row[mid:] for row in A[:mid]]
    A21 = [row[:mid] for row in A[mid:]]
    A22 = [row[mid:] for row in A[mid:]]

    invert_A11 = inverse_matrix(A11, l)
    S22 = subtract(A22, matrix_multiplication(matrix_multiplication(A21, invert_A11, l), A12, l))
    invert_S22 = inverse_matrix(S22, l)
    B11 = matrix_multiplication(invert_A11, add(identity(len(invert_A11)),
                                                matrix_multiplication(matrix_multiplication(A12, invert_S22, l),
                                                                      matrix_multiplication(A21, invert_A11, l), l)), l)
    B12 = matrix_multiplication((-1) * np.array(invert_A11), matrix_multiplication(A12, invert_S22, l), l)
    B21 = matrix_multiplication((-1) * np.array(invert_S22), matrix_multiplication(A21, invert_A11, l), l)
    B22 = invert_S22

    B = [[0 for _ in range(n)] for _ in range(n)]
    for i in range(n // 2):
        for j in range(n // 2):
            B[i][j] = B11[i][j]
            B[i][j + n // 2] = B12[i][j]
            B[i + n // 2][j] = B21[i][j]
            B[i + n // 2][j + n // 2] = B22[i][j]

    return np.array(B)


def invert_matrix_op_count(A, l):
    op_count = 0
    n = len(A)

    if n == 1:
        return 1

    if det(A) == 0:
        return "Macierz nie ma macierzy odwrotnej"

    mid = n // 2
    A11 = [row[:mid] for row in A[:mid]]
    A12 = [row[mid:] for row in A[:mid]]
    A21 = [row[:mid] for row in A[mid:]]
    A22 = [row[mid:] for row in A[mid:]]

    invert_A11 = inverse_matrix(A11, l)
    op_count += invert_matrix_op_count(A11, l)
    tmp_1 = matrix_multiplication(A21, invert_A11, l)
    S22 = subtract(A22, matrix_multiplication(tmp_1, A12, l))
    op_count = op_count + len(A22) ** 2 + matrix_multiplication_operation_count(A21, invert_A11,
                                                                                l) + matrix_multiplication_operation_count(
        tmp_1, A12, l)
    invert_S22 = inverse_matrix(S22, l)
    op_count += invert_matrix_op_count(S22, l)
    tmp_2 = matrix_multiplication(A21, invert_A11, l)
    op_count += matrix_multiplication_operation_count(A21, invert_A11, l)
    tmp_3 = matrix_multiplication(A12, invert_S22, l)
    op_count += matrix_multiplication_operation_count(A12, invert_S22, l)
    tmp_4 = matrix_multiplication(tmp_3, tmp_2, l)
    op_count += matrix_multiplication_operation_count(tmp_3, tmp_2, l)
    tmp_5 = add(identity(len(invert_A11)), tmp_4)
    op_count += len(invert_A11) ** 2
    B11 = matrix_multiplication(invert_A11, tmp_5, l)
    op_count += matrix_multiplication_operation_count(invert_A11, tmp_5, l)
    tmp_6 = matrix_multiplication(A12, invert_S22, l)
    op_count += matrix_multiplication_operation_count(A12, invert_S22, l)
    B12 = matrix_multiplication((-1) * np.array(invert_A11), tmp_6, l)
    op_count += matrix_multiplication_operation_count((-1) * np.array(invert_A11), tmp_6, l)
    tmp_7 = matrix_multiplication(A21, invert_A11, l)
    op_count += matrix_multiplication_operation_count(A21, invert_A11, l)
    B21 = matrix_multiplication((-1) * np.array(invert_S22), tmp_7, l)
    op_count += matrix_multiplication_operation_count((-1) * np.array(invert_S22), tmp_7, l)
    B22 = invert_S22

    op_count = op_count + len(invert_A11) ** 2 + len(invert_S22) ** 2

    return op_count


if __name__ == '__main__':
    M = np.array([[1, 2, 7, 2], [1, 1, 3, 3], [4, 0, 1, 0], [4, 5, 6, 4]])
    print(inverse_matrix(M, 4))
    print(invert_matrix_op_count(M, 4))

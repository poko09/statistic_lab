import numpy as np
import time
import warnings

from matplotlib import pyplot as plt

from mat_mul_op_count import matrix_multiplication_operation_count
from matrix_operations import subtract, matrix_multiplication, add, generate_matrix

def identity(n):
    return np.identity(n)

def det(A):
    return np.linalg.det(A)

def inverse_matrix(A, l):
    n = len(A)

    if n == 1:
        return 1/np.array(A)

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
    op_count = op_count + len(A22)**2 + matrix_multiplication_operation_count(A21, invert_A11, l) + matrix_multiplication_operation_count(tmp_1,A12,l)
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
    op_count += matrix_multiplication_operation_count((-1) * np.array(invert_A11), tmp_6,l)
    tmp_7 = matrix_multiplication(A21, invert_A11, l)
    op_count += matrix_multiplication_operation_count(A21, invert_A11, l)
    B21 = matrix_multiplication((-1) * np.array(invert_S22), tmp_7, l)
    op_count += matrix_multiplication_operation_count((-1) * np.array(invert_S22), tmp_7, l)
    B22 = invert_S22

    op_count = op_count + len(invert_A11)**2 + len(invert_S22)**2

    return op_count


def calculate_time_of_inverting(list_of_matrix):
    time_list = []
    for i in range(len(list_of_matrix)):
        start = time.time()
        inverse_matrix(list_of_matrix[i], 0)
        end = time.time()
        result = round(end-start, 4)
        time_list.append(result)
    return(time_list)


def calculate_number_of_operations(list_of_matrix):
    num_of_operations = []
    for i in range(len(list_of_matrix)):
        num = invert_matrix_op_count(list_of_matrix[i],1)
        num_of_operations.append(num)
    return num_of_operations


def generate_many_matrix():
    list_of_matrix = []
    for i in range(2, 9, 1):
        my_matrix = generate_matrix(2**i)
        list_of_matrix.append(my_matrix)
        warnings.filterwarnings('ignore')
    return list_of_matrix


def draw_chart(points, title_name, y_axis_name):
    x_axis_list = []
    x = 2
    for i in range(len(points)):
        x_axis_list.append(x)
        x +=1
    plt.plot(x_axis_list, points, marker='o')
    plt.title(title_name)
    plt.ylabel(y_axis_name)
    plt.xlabel('rozmiar macierzy 2^n x 2^n')
    plt.show()

if __name__ == '__main__':
    my_list = generate_many_matrix()
    # my_time_list = calculate_time_of_inverting(my_list)
    my_operation_list = calculate_number_of_operations(my_list)

    # draw_chart(my_time_list, "Wykres ze względu na czas", "czas [s]")
    draw_chart(my_operation_list, "Wykres ze względu na liczbę operacji", "liczba operacji")
    print(my_operation_list)





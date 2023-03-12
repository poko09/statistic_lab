import time
import numpy as np
import matplotlib.pyplot as plt

from projekt1.main import generate_matrix, binet, classic_multiplication, matrix_multiplication

count_times_classic = [None for _ in range(17)]
count_operations_classic = [None for _ in range(17)]

count_times_binet = [-1 for _ in range(17)]
count_operations_binet = [-1 for _ in range(17)]


def classic_algorithm_operation_count(A, B):
    op_count = 0
    result = [[0 for _ in range(len(B[0]))] for _ in range(len(A))]
    for i in range(len(A)):
        for j in range(len(B[0])):
            for k in range(len(B)):
                result[i][j] += A[i][k] * B[k][j]
                op_count += 2
    return op_count


def binet_operation_count(A, B):
    op_number = 0

    if len(A[0]) != len(B):
        return 0
    if len(A) != len(A[0]) or len(B) != len(B[0]):
        return 0
    n = len(A)
    if n == 1:
        return 1

    mid = n // 2

    A11 = [row[:mid] for row in A[:mid]]
    A12 = [row[mid:] for row in A[:mid]]
    A21 = [row[:mid] for row in A[mid:]]
    A22 = [row[mid:] for row in A[mid:]]
    B11 = [row[:mid] for row in B[:mid]]
    B12 = [row[mid:] for row in B[:mid]]
    B21 = [row[:mid] for row in B[mid:]]
    B22 = [row[mid:] for row in B[mid:]]

    op_number = op_number + binet_operation_count(A11, B11) + binet_operation_count(A12, B21)
    op_number = op_number + binet_operation_count(A11, B12) + binet_operation_count(A12, B22)
    op_number = op_number + binet_operation_count(A21, B11) + binet_operation_count(A22, B21)
    op_number = op_number + binet_operation_count(A21, B12) + binet_operation_count(A22, B22)
    op_number = op_number + 2 * len(A11) * len(A11[0]) + 2 * len(A21) * len(A21[0])

    return op_number


def matrix_multiplication_operation_count(k, A, B, l):
    if k > l:
        if count_operations_binet[k] == -1:
            return binet_operation_count(A, B)
        else:
            return count_operations_binet[k]
    else:
        if count_operations_classic[k] == -1:
            return classic_algorithm_operation_count(A, B)
        else:
            return count_operations_classic[k]


def time_plot(function, max_size, title):
    k_list = np.arange(2, max_size + 1, 1)
    times = []
    for k in k_list:
        matrix_size = 2 ** k
        A = generate_matrix(matrix_size)
        B = generate_matrix(matrix_size)
        start = time.time()
        function(A, B)
        end = time.time()
        res = end - start
        times.append(res)
        if function.__name__ == "binet":
            count_times_binet[k] = res
        else:
            count_times_classic[k] = res
    fig = plt.subplots()
    plt.plot(k_list, times, 'go')
    plt.xlabel('matrix size 2ᵏ')
    plt.xticks(range(2, max_size + 1))
    plt.ylabel('time [s]')
    plt.title(title)
    plt.savefig("gen_plots/" + function.__name__ + ".png")


def op_plot(function, max_size, title):
    k_list = np.arange(2, max_size + 1, 1)
    op_list = []
    for k in k_list:
        matrix_size = 2 ** k
        A = generate_matrix(matrix_size)
        B = generate_matrix(matrix_size)
        res = function(A, B)
        op_list.append(res)
        if function.__name__ == "binet_operation_count":
            count_operations_binet[k] = res
        else:
            count_operations_classic[k] = res
    fig = plt.subplots()
    plt.plot(k_list, op_list, 'bo')
    plt.xlabel('matrix size 2ᵏ')
    plt.xticks(range(2, max_size + 1))
    plt.ylabel('number of operation')
    plt.title(title)
    plt.savefig("gen_plots/" + function.__name__ + ".png")


def time_plot_from_k(max_size, l, title):
    k_list = np.arange(2, max_size + 1, 1)
    times = []
    for k in k_list:
        matrix_size = 2 ** k
        A = generate_matrix(matrix_size)
        B = generate_matrix(matrix_size)
        if k > l and count_times_binet[k] != -1:
            times.append(count_times_binet[k])
        elif count_times_classic[k] != -1:
            times.append(count_times_classic[k])
        else:
            start = time.time()
            matrix_multiplication(A, B, l)
            end = time.time()
            res = end - start
            times.append(res)
    fig = plt.subplots()
    plt.plot(k_list, times, 'go')
    plt.xlabel('matrix size 2ᵏ')
    plt.xticks(range(2, max_size + 1))
    plt.ylabel('time [s]')
    plt.title(title)
    plt.axvline(x=l, color='r', linestyle='dashed')
    plt.savefig("gen_plots/time_" + str(l) + ".png")


def time_plot_from_l(k, title):
    l_list = np.arange(2, k, 1)
    times = []
    for l in l_list:
        matrix_size = 2 ** k
        A = generate_matrix(matrix_size)
        B = generate_matrix(matrix_size)
        start = time.time()
        matrix_multiplication(A, B, l)
        end = time.time()
        res = end - start
        times.append(res)
    return l_list, times


def op_plot_from_k(max_size, l, title):
    k_list = np.arange(2, max_size + 1, 1)
    op_count = []
    for k in k_list:
        matrix_size = 2 ** k
        A = generate_matrix(matrix_size)
        B = generate_matrix(matrix_size)
        op_count.append(matrix_multiplication_operation_count(k, A, B, l))
    fig = plt.subplots()
    plt.plot(k_list, op_count, 'bo')
    plt.xlabel('matrix size 2ᵏ')
    plt.xticks(range(2, max_size + 1))
    plt.ylabel('number of operations')
    plt.title(title)
    plt.axvline(x=l, color='r', linestyle='dashed')
    plt.savefig("gen_plots/op_" + str(l) + ".png")


def op_plot_from_l(k, title):
    l_list = np.arange(3, k, 1)
    op_count = []
    for l in l_list:
        matrix_size = 2 ** k
        A = generate_matrix(matrix_size)
        B = generate_matrix(matrix_size)
        op_count.append(matrix_multiplication_operation_count(k, A, B, l))
    return l_list, op_count


def generate_time_comp_plots(max_size):
    # time of execution of two (binet's and classic) algorithms
    time_plot(binet, max_size, "Binét's algorithm time of execution")
    time_plot(classic_multiplication, max_size, "Classic algorithm time of execution")

    # time of execution of different k with the same l
    ls = [3, 5, 9]
    for l in ls:
        time_plot_from_k(max_size, l, "Matrix multiplication time with l = {x}".format(x=l))


def generate_operation_comp_plots(max_size):
    # number of operations of two (binet's and classic) algorithms
    op_plot(binet_operation_count, max_size, "Binét's algorithm number of operations")
    op_plot(classic_algorithm_operation_count, max_size, "Classic algorithm number of operations")

    # number of operations of different k with the same l
    ls = [3, 5, 9]
    for l in ls:
        op_plot_from_k(max_size, l, "Matrix multiplication number of operations with l = {x}".format(x=l))


generate_time_comp_plots(10)
generate_operation_comp_plots(10)

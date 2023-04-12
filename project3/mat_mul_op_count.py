def classic_algorithm_operation_count(A, B):
    op_count = 0
    result = [[0 for _ in range(len(B[0]))] for _ in range(len(A))]
    for i in range(len(A)):
        for j in range(len(B[0])):
            for k in range(len(B)):
                result[i][j] += A[i][k] * B[k][j]
                op_count += 2
    return op_count


def binet_operation_count(A, B, l):
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

    if len(A11) > 2 ** l:
        op_number = op_number + binet_operation_count(A11, B11, l) + binet_operation_count(A12, B21, l)
        op_number = op_number + binet_operation_count(A11, B12, l) + binet_operation_count(A12, B22, l)
        op_number = op_number + binet_operation_count(A21, B11, l) + binet_operation_count(A22, B21, l)
        op_number = op_number + binet_operation_count(A21, B12, l) + binet_operation_count(A22, B22, l)
    else:
        op_number = op_number + classic_algorithm_operation_count(A11, B11) + classic_algorithm_operation_count(A12, B21)
        op_number = op_number + classic_algorithm_operation_count(A11, B12) + classic_algorithm_operation_count(A12, B22)
        op_number = op_number + classic_algorithm_operation_count(A21, B11) + classic_algorithm_operation_count(A22, B21)
        op_number = op_number + classic_algorithm_operation_count(A21, B12) + classic_algorithm_operation_count(A22, B22)

    op_number = op_number + 2 * len(A11) * len(A11[0]) + 2 * len(A21) * len(A21[0])

    return op_number


def matrix_multiplication_operation_count(A, B, l):
    if len(A) > 2**l:
        return binet_operation_count(A, B, l)
    else:
        return classic_algorithm_operation_count(A, B)
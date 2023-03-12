import time
import numpy as np

'''
    All multiplying methods:
'''


def classic_multiplication(matrix1, matrix2):
    result = [[0 for _ in range(len(matrix2[0]))] for _ in range(len(matrix1))]
    for i in range(len(matrix1)):
        for j in range(len(matrix2[0])):
            for k in range(len(matrix2)):
                result[i][j] += matrix1[i][k] * matrix2[k][j]
    return (result)


def strassen(A, B):
    n = len(A)
    if n == 1:
        return [[A[0][0] * B[0][0]]]

    # podział macierzy na 4 'podmacierze' - bloki
    mid = n // 2

    A11 = [row[:mid] for row in A[:mid]]
    A12 = [row[mid:] for row in A[:mid]]
    A21 = [row[:mid] for row in A[mid:]]
    A22 = [row[mid:] for row in A[mid:]]
    B11 = [row[:mid] for row in B[:mid]]
    B12 = [row[mid:] for row in B[:mid]]
    B21 = [row[:mid] for row in B[mid:]]
    B22 = [row[mid:] for row in B[mid:]]

    # Utwórz 7 wartości pomocniczych (Oblicz rekurencyjnie siedem iloczynów)

    P1 = strassen(A11, subtract(B12, B22))
    P2 = strassen(add(A11, A12), B22)
    P3 = strassen(add(A21, A22), B11)
    P4 = strassen(A22, subtract(B21, B11))
    P5 = strassen(add(A11, A22), add(B11, B22))
    P6 = strassen(subtract(A12, A22), add(B21, B22))
    P7 = strassen(subtract(A11, A21), add(B11, B12))

    # Obliczanie wartości macierzy na podstawie 7 składowych
    C11 = subtract(add(add(P5, P4), P6), P2)
    C12 = add(P1, P2)
    C21 = add(P3, P4)
    C22 = subtract(subtract(add(P5, P1), P3), P7)

    # złożenie macierzy do całości
    result = [[0 for j in range(n)] for i in range(n)]
    for i in range(mid):
        for j in range(mid):
            result[i][j] = C11[i][j]
            result[i][j + mid] = C12[i][j]
            result[i + mid][j] = C21[i][j]
            result[i + mid][j + mid] = C22[i][j]

    return result


def binet(A, B):
    # Sprawdzenie czy liczba kolumn macierzy A jest równa liczbie wierszy macierzy B
    if len(A[0]) != len(B):
        return None

    # Sprawdzenie czy macierze A i B są macierzami kwadratowymi
    if len(A) != len(A[0]) or len(B) != len(B[0]):
        return None

    n = len(A)

    # Jeśli n = 1 to zwróć iloczyn pojedynczych elementów
    if n == 1:
        return [[A[0][0] * B[0][0]]]

    # podział macierzy na 4 'podmacierze' - bloki
    mid = n // 2

    A11 = [row[:mid] for row in A[:mid]]
    A12 = [row[mid:] for row in A[:mid]]
    A21 = [row[:mid] for row in A[mid:]]
    A22 = [row[mid:] for row in A[mid:]]
    B11 = [row[:mid] for row in B[:mid]]
    B12 = [row[mid:] for row in B[:mid]]
    B21 = [row[:mid] for row in B[mid:]]
    B22 = [row[mid:] for row in B[mid:]]

    # Wywołanie rekurencyjne
    C11 = add(binet(A11, B11), binet(A12, B21))
    C12 = add(binet(A11, B12), binet(A12, B22))
    C21 = add(binet(A21, B11), binet(A22, B21))
    C22 = add(binet(A21, B12), binet(A22, B22))

    # Połączenie czterech podmacierzy w jedną macierz wynikową
    C = [[0 for j in range(n)] for i in range(n)]
    for i in range(n // 2):
        for j in range(n // 2):
            C[i][j] = C11[i][j]
            C[i][j + n // 2] = C12[i][j]
            C[i + n // 2][j] = C21[i][j]
            C[i + n // 2][j + n // 2] = C22[i][j]

    return C


"""
    Multiplication program:
"""


def matrix_multiplication(A, B, l):
    if len(A) > 2 ** l:
        return binet(A, B)
    else:
        return classic_multiplication(A, B)

"""
    Helper function:
"""


def add(A, B):
    return [[A[i][j] + B[i][j] for j in range(len(A[0]))] for i in range(len(A))]


def subtract(A, B):
    return [[A[i][j] - B[i][j] for j in range(len(A[0]))] for i in range(len(A))]


def generate_matrix(n):
    array = np.random.uniform(5, 20, size=(n, n))
    return array


if __name__ == '__main__':
    matrix1 = generate_matrix(4)
    matrix2 = generate_matrix(4)

    start = time.time()
    print(binet(matrix1, matrix2))
    end = time.time()
    print(end - start)

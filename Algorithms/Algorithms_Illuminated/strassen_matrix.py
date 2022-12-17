import numpy as np


def strassen(mat1: np.array, mat2: np.array):
    if len(mat1) == 1:
        return mat1 * mat2
    
    a, b, c, d = subdivide(mat1)
    e, f, g, h = subdivide(mat2)

    p1 = strassen(a, f - h)
    p2 = strassen(a + b, h)
    p3 = strassen(c + d, e)
    p4 = strassen(d, g - e)
    p5 = strassen(a + d, e + h)
    p6 = strassen(b - d, g + h)
    p7 = strassen(a - c, e + f)

    tl = p5 + p4 - p2 + p6
    tr = p1 + p2
    bl = p3 + p4
    br = p1 + p5 - p3 - p7

    mat = np.vstack((np.hstack((tl, tr)), np.hstack((bl, br))))
    return mat


def subdivide(mat: np.array):
    row_divide = len(mat) // 2
    col_divide = len(mat[0]) // 2

    return mat[:row_divide, :col_divide], mat[:row_divide, col_divide:], mat[row_divide:, :col_divide], mat[row_divide:, col_divide:]


array1 = np.array([[3, 7], [4, 9]])
array2 = np.array([[6, 2], [5, 8]])
print(strassen(array1, array2))
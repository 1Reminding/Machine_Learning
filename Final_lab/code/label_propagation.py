import math
import random
import numpy as np
from collections import defaultdict
from scipy.sparse import coo_matrix


class matCoo:
    class elems:
        def __init__(self, v, row, col):
            self.v = v
            self.row = row
            self.col = col

        def __lt__(self, other):
            if isinstance(other, matCoo.elems):
                return self.row < other.row or (self.row == other.row and self.col < other.col)
            elif isinstance(other, int):
                return self.row < other
            return False

    def __init__(self, n0=0, m0=0):
        self.elem = []
        self.n = n0
        self.m = m0
        self.totalElements = 0

    def createMat(self, n0, m0):
        self.n = n0
        self.m = m0
        self.totalElements = 0
        self.elem = []

    def matTimes(self, c):
        if c == 0:
            self.elem = []
            self.n = 0
            self.m = 0
            self.totalElements = 0
            return
        for e in self.elem:
            e.v *= c

    def append(self, n0, m0, val):
        if abs(val) < 1e-10:
            return
        self.elem.append(matCoo.elems(val, n0, m0))
        self.totalElements += 1


class mat:
    def __init__(self, n0=1, m0=1):
        self.n = n0
        self.m = m0
        self.v = np.zeros((n0, m0), dtype=np.float64)

    def createMat(self, n0, m0):
        self.n = n0
        self.m = m0
        self.v = np.zeros((n0, m0), dtype=np.float64)

    def matTimes(self, c):
        self.v *= c

    def findDiff(self, other):
        if self.n != other.n or self.m != other.m:
            return -1
        return np.sum(np.abs(self.v - other.v))

    def getval(self, x, y):
        return self.v[x, y]

    def editval(self, x, y, val):
        self.v[x, y] = val

    def setneg(self):
        self.v.fill(-1.0)

    def editval2(self, x, y):
        self.v[x, :] = 0.0
        self.v[x, y] = 1.0


def matCoo_to_scipy_coo(mat_coo):
    data = [e.v for e in mat_coo.elem]
    row = [e.row for e in mat_coo.elem]
    col = [e.col for e in mat_coo.elem]
    return coo_matrix((data, (row, col)), shape=(mat_coo.n, mat_coo.m))


def matMultiply(W_sparse, X_dense, Y_dense):
    return W_sparse.dot(X_dense)


def dataProcess(y_old, y_new, preserved=0.8, changed=0.1, masked=0.1):
    n0, m0 = y_old.n, y_old.m
    y_new.createMat(n0, m0)
    y_new.setneg()

    for i in range(y_old.n):
        if y_old.v[i, 0] != -1:
            r = random.random()
            if r < preserved:
                y_new.v[i, :] = y_old.v[i, :]
            elif preserved < r < preserved + changed:
                sd = random.randint(0, y_old.m - 1)
                while y_old.v[i, sd] == 1:
                    sd = random.randint(0, y_old.m - 1)
                y_new.v[i, :] = 0
                y_new.v[i, sd] = 1
            else:
                y_new.v[i, :] = -1


def rectify(W, y_label, y_pred, y_res):
    y_res.createMat(y_pred.n, y_pred.m)
    y_res.v[:] = y_pred.v

    for i in range(y_label.n):
        if y_label.v[i, 0] != -1:
            p = defaultdict(int)
            for j in range(W.indptr[i], W.indptr[i + 1]):  # 使用 indptr 访问
                col = W.indices[j]
                p[int(y_pred.v[col, 0])] += 1

            max_class = max(p, key=p.get)
            y_res.v[i, 0] = max_class


def labelPropagation(X, y_label, y_pred, y_res, alpha=0.5, max_iter=1000):
    n_samples = X.n
    n_classes = y_label.m

    Y = mat(n_samples, n_classes)
    Y.createMat(n_samples, n_classes)
    Y.v[:] = y_label.v

    W_sparse = matCoo_to_scipy_coo(X)
    normalized_values = W_sparse.data / np.max(W_sparse.data)
    W_sparse.data = np.exp(-alpha * normalized_values ** 2)
    W_sparse = W_sparse.tocsr()  # 转换为 CSR 格式

    Y_new = np.zeros((n_samples, n_classes), dtype=np.float64)
    for iteration in range(max_iter):
        Y_old = Y.v.copy()
        Y_new = matMultiply(W_sparse, Y.v, Y_new)
        row_sums = np.sum(Y_new, axis=1)
        row_sums[row_sums == 0] = 1
        Y_new /= row_sums[:, np.newaxis]
        mask = (y_label.v == -1)
        Y.v[mask] = Y_new[mask]
        diff = np.sum(np.abs(Y_old - Y.v)) / n_samples
        if diff < 1e-5:
            break

    y_pred.createMat(n_samples, 1)
    y_pred.v[:, 0] = np.argmax(Y.v, axis=1)
    rectify(W_sparse, y_label, y_pred, y_res)

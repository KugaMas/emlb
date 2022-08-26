import math
import time
import numba as nb
import numpy as np

from scipy import spatial
from scipy.optimize import leastsq
from abc import ABC, abstractmethod

class EventDenoisors(ABC):
    def __init__(self, size_x, size_y):
        super().__init__()
        self.sizeX = size_x
        self.sizeY = size_y


    @abstractmethod
    def run(self, ts, x, y, p):
        pass


class gef(EventDenoisors):
    def __init__(self, size_x, size_y, param):
        super().__init__(size_x, size_y)

    def run(self, ts, x, y, p):
        pass


class evflow(EventDenoisors):
    def __init__(self, size_x, size_y, param):
        super().__init__(size_x, size_y)
        self.thres = param[0]
        self.round = param[1]
        self.ev_array = None
        
    def run(self, ts, x, y, p):
        self.ev_array = np.c_[x, y, (ts - ts[0]) * 1E-3]
        tree = spatial.KDTree(self.ev_array)
        idxIn = tree.query_ball_point(self.ev_array, r=self.round, p=np.inf)

        st = time.time()
        mask_2 = self.calVelocity(x, y, ts, idxIn)
        ed = time.time()
        print(ed - st)

        # st = time.time()
        # calVelocity = np.vectorize(self.fitLeastSquare)
        # v = calVelocity(idxIn)
        # mask = np.array(v) < self.thres
        # ed = time.time()
        # print(ed - st)

        return mask_2

    # # @nb.jit()
    def calVelocity(self, x, y, ts, idxIn):
        result = np.zeros((ts.shape)).astype(np.bool_)
        for i, idx in enumerate(idxIn):
            if len(idx) < 3:
                result[i] = False

            else:
                # least square fitting
                A = np.c_[x[idx], y[idx], np.ones(ts[idx].shape)]
                b = ts[idx]
                X = np.dot(np.linalg.pinv(np.dot(A.T, A)), np.dot(A.T, b))

                a = X[0]
                b = X[1]
                c = -1
                d = X[2]

                result[i] = math.sqrt((c / (a + np.spacing(1))) ** 2 + (c / (b + np.spacing(1))) ** 2) < self.thres

        return result

    
    # def least_square(self, A, b):
    #     X = np.dot(np.linalg.inv(np.dot(A.T, A)), np.dot(A.T, b))
    #     return X


    def fitLeastSquare(self, idxIn):
        if len(idxIn) < 3:
            return np.inf
        else:
            x = self.ev_array[idxIn][:, 0]
            y = self.ev_array[idxIn][:, 1]
            t = self.ev_array[idxIn][:, 2]

            # 构建方程式
            A = np.c_[x, y, np.ones(t.shape)]
            b = np.expand_dims(np.array(t), 1)

            # 最小二乘拟合
            X, *_ = np.linalg.lstsq(A, b, rcond=None)

            # 计算速度
            a = X[0]
            b = X[1]
            c = -1
            d = X[2]

            return math.sqrt((c / (a + np.spacing(1))) ** 2 + (c / (b + np.spacing(1))) ** 2)

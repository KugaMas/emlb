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

        calVelocity = np.vectorize(self.fitLeastSquare)
        v = calVelocity(idxIn)

        return np.array(v) < self.thres

    def fitLeastSquare(self, idxIn):
        if len(idxIn) < 3:
            return np.inf
        else:
            x = self.ev_array[idxIn][:, 0]
            y = self.ev_array[idxIn][:, 1]
            t = self.ev_array[idxIn][:, 2]

            # least square fitting
            A = np.c_[x, y, np.ones(t.shape)]
            b = np.expand_dims(np.array(t), 1)
            X, *_ = np.linalg.lstsq(A, b, rcond=None)

            # 计算速度
            a, b, c, d = X[0], X[1], -1, X[2]

            return math.sqrt((c / (a + np.spacing(1))) ** 2 + (c / (b + np.spacing(1))) ** 2)

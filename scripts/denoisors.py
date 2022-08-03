import os
import os.path as osp
import numpy as np
from abc import ABC, abstractmethod

import utils.cdn_utils as cdn
import utils.event_utils as ute

cwd = os.getcwd() + '/models'


class EventDenoisors(ABC):
    def __init__(self, size, use_polarity=True, excl_hotpixel=True):
        super().__init__()
        self.name           = 'Template'
        self.annotation     = 'Template'
        self.use_polarity   = use_polarity
        self.excl_hotpixel  = excl_hotpixel

        self.size  = size
        self.model = None

    @staticmethod
    def pre_prosess(self, ev, size):
        if self.use_polarity is False:
            ev = ute.polarity_removal(ev, size)
        if self.excl_hotpixel is False:
            ev = ute.hotpixel_removal(ev, size)

        ts, x, y, p = np.split(ev, [1, 2, 3], axis=1)
        ts = ts.astype(np.int64)
        x = x.astype(np.uint16)
        y = y.astype(np.uint16)
        p = p.astype(np.bool_)

        return ts, x, y, p
    
    @abstractmethod
    def run(self, ev, fr):
        pass


class dwf(EventDenoisors):
    def __init__(self, use_polarity=True, excl_hotpixel=True,
                 threshold=1, radius=10, double_mode=True, window_size=8):
        self.name           = 'DWF'
        self.annotation     = 'Double Window Filter'
        self.use_polarity   = use_polarity
        self.excl_hotpixel  = excl_hotpixel
            
        self.threshold = threshold
        self.radius    = radius
        self.duoMode   = double_mode
        self.winSize   = window_size

        self.params = {
            'threshold': threshold,
            'radius'   : radius,
            'duoMode'  : double_mode,
            'winSize'  : window_size,
        }

    def run(self, ev, fr, size):
        ts, x, y, p = self.pre_prosess(self, ev, size)
        model = cdn.dwf(size[0], size[1], tuple(self.params.values()))
        idx = model.run(ts, x, y, p)
        return ev[idx]


class mlpf(EventDenoisors):
    def __init__(self, use_polarity=True, excl_hotpixel=True,
                 threshold=0.5, radius=3, tau_ts=1E5, cal_timestamp=True, cal_polarity=True, model_path="2xMSEO1H20_linear_7.pt"):
        self.name           = 'MLPF'
        self.annotation     = 'Multi Layer Perceptron Filter'
        self.use_polarity   = use_polarity
        self.excl_hotpixel  = excl_hotpixel

        self.params = {
            'threshold': threshold,
            'radius'   : radius,
            'tauTs'    : tau_ts,
            'cal_timestamp' : cal_timestamp,
            'cal_polarity'  : cal_polarity,
            'model_path'    : osp.join(cwd, model_path),
        }

    def run(self, ev, fr, size):
        ts, x, y, p = self.pre_prosess(self, ev, size)
        model = cdn.mlpf(size[0], size[1], tuple(self.params.values()))
        idx = model.run(ts, x, y, p)
        return ev[idx]

def Denoisor(idx, args):
    model = eval(args.denoisors[idx].lower())
    return model(args.use_polarity, args.excl_hotpixel, *args.params[idx])

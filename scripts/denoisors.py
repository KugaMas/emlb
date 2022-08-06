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


class baf(EventDenoisors):
    def __init__(self, use_polarity=True, excl_hotpixel=True,
                 threshold=1, 
                 radius_norm_l2=1, 
                 delta_t=10000, 
                 cal_polarity=True):

        self.name           = 'BAF'
        self.annotation     = 'Background Activity Filter'
        self.use_polarity   = use_polarity
        self.excl_hotpixel  = excl_hotpixel

        self.params = {
            'threshold': threshold,
            'radiusNL2': radius_norm_l2,
            'deltaT'   : delta_t,
            'cal_polarity' : cal_polarity,
        }

    def run(self, ev, fr, size):
        ts, x, y, p = self.pre_prosess(self, ev, size)
        model = cdn.baf(size[0], size[1], tuple(self.params.values()))
        idx = model.run(ts, x, y, p)
        return ev[idx]


class dwf(EventDenoisors):
    def __init__(self, use_polarity=True, excl_hotpixel=True,
                 threshold=1, 
                 radius_norm_l1=10, 
                 window_size=8, 
                 double_mode=True):

        self.name           = 'DWF'
        self.annotation     = 'Double Window Filter'
        self.use_polarity   = use_polarity
        self.excl_hotpixel  = excl_hotpixel

        self.params = {
            'threshold': threshold,
            'radiusNL1': radius_norm_l1,
            'winSize'  : window_size,
            'duoMode'  : double_mode,
        }

    def run(self, ev, fr, size):
        ts, x, y, p = self.pre_prosess(self, ev, size)
        model = cdn.dwf(size[0], size[1], tuple(self.params.values()))
        idx = model.run(ts, x, y, p)
        return ev[idx]


class mlpf(EventDenoisors):
    def __init__(self, use_polarity=True, excl_hotpixel=True,
                 threshold=0.5, 
                 radius_norm_l2=3, 
                 tau_ts=1E5, 
                 batch_size=1E7, 
                 cal_timestamp=True, 
                 cal_polarity=True, 
                 model_path="MLPF_2xMSEO1H20_linear_7.pt"):
                 
        self.name           = 'MLPF'
        self.annotation     = 'Multi Layer Perceptron Filter'
        self.use_polarity   = use_polarity
        self.excl_hotpixel  = excl_hotpixel

        self.params = {
            'threshold' : threshold,
            'radiusNL2' : radius_norm_l2,
            'tauTs'     : tau_ts,
            'batch_size': int(batch_size),
            'cal_timestamp' : cal_timestamp,
            'cal_polarity'  : cal_polarity,
            'model_path'    : osp.join(cwd, model_path),
        }

    def run(self, ev, fr, size):
        ts, x, y, p = self.pre_prosess(self, ev, size)
        model = cdn.mlpf(size[0], size[1], tuple(self.params.values()))
        idx = model.run(ts, x, y, p)
        return ev[idx]


class edncnn(EventDenoisors):
    def __init__(self, use_polarity=True, excl_hotpixel=True,
                 threshold=0.5, 
                 radius_norm_l2=12,
                 depth=2,
                 batch_size=750,
                 model_path="EDnCNN_all_trained_v9.pt"):
                 
        self.name           = 'EDnCNN'
        self.annotation     = 'Event Denoise Convolutional Neural Network'
        self.use_polarity   = use_polarity
        self.excl_hotpixel  = excl_hotpixel

        self.params = {
            'threshold'  : threshold,
            'radiusNL2'  : radius_norm_l2,
            'depth'      : depth,
            'batch_size' : int(batch_size),
            'model_path' : osp.join(cwd, model_path),
        }

    def run(self, ev, fr, size):
        ts, x, y, p = self.pre_prosess(self, ev, size)
        model = cdn.edncnn(size[0], size[1], tuple(self.params.values()))
        idx = model.run(ts, x, y, p)
        return ev[idx]


def Denoisor(idx, args):
    model = eval(args.denoisors[idx].lower())
    return model(args.use_polarity, args.excl_hotpixel, *args.params[idx])

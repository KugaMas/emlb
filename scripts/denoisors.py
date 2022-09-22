import os
import math
import os.path as osp
import numpy as np
from abc import ABC, abstractmethod

import utils.edn_utils as edn
import utils.cdn_utils as cdn
import utils.event_utils as ute

cwd = os.getcwd() + '/models'


class EventDenoisors(ABC):
    def __init__(self, use_polarity=True, excl_hotpixel=True):
        super().__init__()
        self.name           = 'Template'
        self.annotation     = 'Template'
        self.use_polarity   = use_polarity
        self.excl_hotpixel  = excl_hotpixel

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


class raw(EventDenoisors):
    def __init__(self, use_polarity=True, excl_hotpixel=True):
        self.name           = 'Raw'
        self.annotation     = 'Raw Event Data'
        self.use_polarity   = use_polarity
        self.excl_hotpixel  = excl_hotpixel

        self.params = None

    def run(self, ev, fr, size):
        ts, x, y, p = self.pre_prosess(self, ev, size)
        return ev


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


class nn(EventDenoisors):
    def __init__(self, use_polarity=True, excl_hotpixel=True,
                 threshold=1, 
                 radius_norm_l2=1, 
                 delta_t=10000,
                 refractory_t=3000,
                 cal_polarity=True):

        self.name           = 'NN'
        self.annotation     = 'Nearest Neighbor'
        self.use_polarity   = use_polarity
        self.excl_hotpixel  = excl_hotpixel

        self.params = {
            'threshold': threshold,
            'radiusNL2': radius_norm_l2,
            'deltaT'   : delta_t,
            'refractoryT'  : refractory_t,
            'cal_polarity' : cal_polarity,
        }

    def run(self, ev, fr, size):
        ts, x, y, p = self.pre_prosess(self, ev, size)
        model = cdn.nn(size[0], size[1], tuple(self.params.values()))
        idx = model.run(ts, x, y, p)
        return ev[idx]


class knoise(EventDenoisors):
    def __init__(self, use_polarity=True, excl_hotpixel=True,
                 supportors=1,
                 delta_t=10000):

        self.name           = 'KNoise'
        self.annotation     = 'Khodamoradi Noise'
        self.use_polarity   = use_polarity
        self.excl_hotpixel  = excl_hotpixel

        self.params = {
            'supportors': supportors,
            'deltaT'   : delta_t,
        }

    def run(self, ev, fr, size):
        ts, x, y, p = self.pre_prosess(self, ev, size)
        model = cdn.knoise(size[0], size[1], tuple(self.params.values()))
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


class evflow(EventDenoisors):
    def __init__(self, use_polarity=True, excl_hotpixel=True,
                 velocity_th = 10,
                 radius_norm_l2 = 1,
                 delta_t=1500):
        self.name           = 'EvFlow'
        self.annotation     = 'Event Flow Filter'
        self.use_polarity   = use_polarity
        self.excl_hotpixel  = excl_hotpixel

        self.params = {
            'threshold': velocity_th,
            'radiusNL2': radius_norm_l2,
            'deltaT'    : delta_t,
        }

    def run(self, ev, fr, size):
        ts, x, y, p = self.pre_prosess(self, ev, size)
        model = cdn.evflow(size[0], size[1], tuple(self.params.values()))
        idx = model.run(ts, x, y, p)
        return ev[idx]


class ynoise(EventDenoisors):
    def __init__(self, use_polarity=True, excl_hotpixel=True,
                 supportors=2,
                 distance_l2_norm=1,
                 delta_t=10000):

        self.name           = 'YNoise'
        self.annotation     = 'Yang Noise'
        self.use_polarity   = use_polarity
        self.excl_hotpixel  = excl_hotpixel

        self.params = {
            'supportors': supportors,
            'distanceL2': distance_l2_norm,
            'deltaT'    : delta_t,
        }

    def run(self, ev, fr, size):
        ts, x, y, p = self.pre_prosess(self, ev, size)
        model = cdn.ynoise(size[0], size[1], tuple(self.params.values()))
        idx = model.run(ts, x, y, p)
        return ev[idx]


class timesurface(EventDenoisors):
    def __init__(self, use_polarity=True, excl_hotpixel=True,
                 threshold=0.4, 
                 radius_norm_l2=1,  
                 decay=30000):
                 
        self.name           = 'TS'
        self.annotation     = 'Time Surface'
        self.use_polarity   = use_polarity
        self.excl_hotpixel  = excl_hotpixel

        self.params = {
            'threshold' : threshold,
            'radiusNL2' : radius_norm_l2,
            'decay'     : decay,
            'deltaTNeg' : 0,
            'deltaTPos' : 0,
        }

    def run(self, ev, fr, size):
        ts, x, y, p = self.pre_prosess(self, ev, size)
        model = cdn.timesurface(size[0], size[1], tuple(self.params.values()))
        idx = model.run(ts, x, y, p)
        return ev[idx]


class fsae(EventDenoisors):
    def __init__(self, use_polarity=True, excl_hotpixel=True,
                 threshold=0.4, 
                 radius_norm_l2=1,  
                 decay=30000,
                 delta_t_neg=5000):
                 
        self.name           = 'FSAE'
        self.annotation     = 'Filtered Surface of Active Events'
        self.use_polarity   = use_polarity
        self.excl_hotpixel  = excl_hotpixel

        self.params = {
            'threshold' : threshold,
            'radiusNL2' : radius_norm_l2,
            'decay'     : decay,
            'deltaTNeg' : delta_t_neg,
            'deltaTPos' : 0,
        }

    def run(self, ev, fr, size):
        ts, x, y, p = self.pre_prosess(self, ev, size)
        model = cdn.timesurface(size[0], size[1], tuple(self.params.values()))
        idx = model.run(ts, x, y, p)
        return ev[idx]


class iets(EventDenoisors):
    def __init__(self, use_polarity=True, excl_hotpixel=True,
                 threshold=0.5, 
                 radius_norm_l2=1,  
                 decay=30000,
                 delta_t_neg=5000,
                 delta_t_pos=5000):
                 
        self.name           = 'IETS'
        self.annotation     = 'Inceptive Event Time Surfaces'
        self.use_polarity   = use_polarity
        self.excl_hotpixel  = excl_hotpixel

        self.params = {
            'threshold' : threshold,
            'radiusNL2' : radius_norm_l2,
            'decay'     : decay,
            'deltaTNeg' : delta_t_neg,
            'deltaTPos' : delta_t_pos,
        }

    def run(self, ev, fr, size):
        ts, x, y, p = self.pre_prosess(self, ev, size)
        model = cdn.timesurface(size[0], size[1], tuple(self.params.values()))
        idx = model.run(ts, x, y, p)
        return ev[idx]


class gef(EventDenoisors):
    def __init__(self, use_polarity=True, excl_hotpixel=True):
        self.name       = 'GEF'
        self.annotation = 'Guided Event Filter'
        self.use_polarity   = use_polarity
        self.excl_hotpixel  = excl_hotpixel

        self.params ={
            'TODO': False,
        }

    def run(self, ev, fr, size):
        ts, x, y, p = self.pre_prosess(self, ev, size)
        return ev


class mlpf(EventDenoisors):
    def __init__(self, use_polarity=True, excl_hotpixel=True,
                 threshold=0.5, 
                 radius_norm_l2=3, 
                 tau_ts=1E5, 
                 batch_size=1000, 
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
                 batch_size=100,
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


class evzoom(EventDenoisors):
    def __init__(self, use_polarity=True, excl_hotpixel=True):
        self.name       = 'EvZoom'
        self.annotation = 'Event Zoom'
        self.use_polarity   = use_polarity
        self.excl_hotpixel  = excl_hotpixel

        self.params ={
            'TODO': False,
        }

    def run(self, ev, fr, size):
        ts, x, y, p = self.pre_prosess(self, ev, size)
        return ev


def Denoisor(idx, args):
    model = eval(args.denoisors[idx].lower())
    
    if idx >= len(args.params):
        return model(args.use_polarity, args.excl_hotpixel)

    return model(args.use_polarity, args.excl_hotpixel, *args.params[idx])

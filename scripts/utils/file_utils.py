import os
import pickle

import numpy as np
import pandas as pd
import os.path as osp

from dv import AedatFile
from zipfile import ZipFile
from scipy.io import savemat
from numpy.lib import recfunctions as rfn


def load_file(file_path, size=(-1, -1), aps=False):
    ev, fr = None, None
    ext = osp.splitext(osp.basename(file_path))[1]
    assert ext in ['.h5', '.txt', '.pkl', '.aedat4'], "Unsupported read file type"
    
    if ext == '.aedat4':
        with AedatFile(file_path) as f:
            ev = np.hstack([packet for packet in f['events'].numpy()])
            ev = rfn.structured_to_unstructured(ev)[1:, :4].astype(np.uint64)
    if ext == '.txt':
        with open(file_path, "r+") as f:
            ev = pd.read_csv(f, sep='\s+', skiprows=[0], header=None,
                             dtype={'0': np.float32, '1': np.int8, '2': np.int8, '3': np.int8})
            ev = np.array(ev).astype(np.uint64)
    if ext == '.pkl':
        with open(file_path, "rb+") as f:
            ev = pd.read_pickle(f)['events']
            ev = np.array(ev).astype(np.uint64)[1:, :]
    if ext == '.zip':
        with ZipFile(file_path, 'r').open(f'events.txt') as f:  # 改events.txt
            ev = pd.read_pickle(f)
            ev = np.array(ev).astype(np.uint64)[1:, :]

    if size == (-1, -1):
        x = int(np.max(ev[:, 1]) + 1)
        y = int(np.max(ev[:, 2]) + 1)
        size = (x, y)

    if aps:
        if ext == '.aedat4':
            with AedatFile(file_path) as f:
                fr = [[np.array(packet.image).squeeze(), packet.timestamp] for packet in f['frames']]
        if ext == '.pkl':
            with open(file_path, "rb+") as f:
                fr = [[np.array(packet[0]), packet[1]] for packet in pd.read_pickle(f)['frames']]

    return ev, fr, size


def save_file(ev, fr, params, file_path):
    if params is None: return

    ext = osp.splitext(osp.basename(file_path))[1]
    assert ext in ['.h5', '.pkl', '.txt'], "Unsupported write file type"

    dir_path, file_name = osp.split(file_path)
    if not osp.exists(dir_path):
        os.makedirs(dir_path)
        os.makedirs(f"{dir_path}/.params")

    # save paramters
    savemat(f"{dir_path}/.params/{file_name.replace(ext, '.mat')}", params)

    if ext == '.pkl':
        with open(file_path, 'wb+') as f:
            pickle.dump(dict(events=ev, frames=fr), f)

    if ext == '.txt':
        with open(file_path, 'wt+') as f:
            np.savetxt(f, ev, fmt="%16d %3d %3d %1d", delimiter=' ', newline='\n')

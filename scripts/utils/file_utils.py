import os
import pickle
import zipfile

import numpy as np
import pandas as pd
import os.path as osp

from dv import AedatFile
from zipfile import ZipFile
from scipy.io import savemat
from numpy.lib import recfunctions as rfn


def load_file(file_path, aps=False):
    ev, fr, size = None, None, None
    ext = osp.splitext(osp.basename(file_path))[1]
    assert ext in ['.h5', '.txt', '.pkl', '.zip', '.aedat4'], "Unsupported read file type"
    
    if ext == '.aedat4':
        with AedatFile(file_path) as f:
            ev = np.hstack([packet for packet in f['events'].numpy()])
            ev = rfn.structured_to_unstructured(ev)[1:, :4].astype(np.uint64)
            size = f['events'].size[::-1]
    if ext == '.txt':
        with open(file_path, "r+") as f:
            ev = pd.read_csv(f, sep='\s+', header=None, skiprows=[0],
                             dtype={'0': np.float32, '1': np.int8, '2': np.int8, '3': np.int8})
            ev = np.array(ev).astype(np.uint64)
        with open(file_path, "r+") as f:
            size = tuple(np.loadtxt(f, max_rows=1).astype(np.int_))
    if ext == '.pkl':
        with open(file_path, "rb+") as f:
            ev = pd.read_pickle(f)['events']
            ev = np.array(ev).astype(np.uint64)
        with open(file_path, "rb+") as f:
            size = pd.read_pickle(f)['size']

    if size == None:
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


def save_file(ev, fr, size, params, file_path):
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
            pickle.dump(dict(events=ev, frames=fr, size=size), f)

    if ext == '.txt':
        with open(file_path, 'wt') as f:
            f.write('%3d %3d\n' % (size[0], size[1]))
        with open(file_path, 'at') as f:
            np.savetxt(f, ev, fmt="%16d %3d %3d %1d", delimiter=' ', newline='\n')


def search_file(args, model, dataset, seq):
    if model.name.lower() == 'raw': 
        return seq.path, True, False

    search_path = f"{args.output_path}/{model.name}"
    search_name = f"{seq.name}.{args.output_file_type}"
    
    for root, dirs, files in os.walk(search_path):
        for name in files:
            if name == search_name: 
                return osp.join(root, name), True, args.replace_file

    output_name = f"{dataset.name}/{seq.category}/{seq.name}.{args.output_file_type}"
    
    return osp.join(search_path, output_name), False, args.replace_file

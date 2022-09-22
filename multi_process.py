import os
import time
import argparse
import threading
import numpy as np
import multiprocessing

from options.options import set_inference_options, Database
from scripts.denoisors import Denoisor
from scripts.utils.file_utils import load_file, save_file, search_file
from scripts.utils.event_utils import calc_event_structural_ratio, pack_along_timestamp, projection_image 
from tqdm import tqdm


def Func(seq):
    output_path, search_flag, replace_flag = search_file(args, model, dataset, seq)
    # skip existing files
    if search_flag and not replace_flag:
        if not args.calc_esr_score: return [seq, np.nan]
        # load denoised event data
        ev, fr, size = load_file(output_path, aps=seq.use_aps)
    else:
        # load noisy event data and perform inference
        ev, fr, size = load_file(seq.path, aps=seq.use_aps)
        ev = model.run(ev, fr, size)
        # save inference result
        if args.save_file:
            save_file(ev, fr, size, model.params, output_path)

    # calculate ESR
    score = calc_event_structural_ratio(ev, size)
    return [seq, score]


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Deployment of EMLB benchmark')
    parser.add_argument('-i', '--input_path', type=str, default='./datasets', help='path to load dataset')
    parser.add_argument('-o', '--output_path', type=str, default='./results', help='path to output dataset')
    parser.add_argument('-d', '--denoisors', type=list, default=['raw', 'baf', 'nn', 'knoise', 'dwf', 'evflow', 'ynoise', 'timesurface', 'iets', 'mlpf', 'edncnn'], help='choose denoisors')
    parser.add_argument("-p", "--params", type=float, default=[], nargs='+', help="specified parameters")

    parser.add_argument('-w', '--save_file', action='store_false', help="save denoising result")
    parser.add_argument('-s', '--calc_esr_score', action='store_false', help="ecaluate esr performance")
    parser.add_argument('-r', '--replace_file', action='store_true', help="replace the former output file")

    parser.add_argument('--process', type=int, default=64)

    args = set_inference_options(parser)

    for dataset in Database(args):
        table = dataset.table()
        pbar  = tqdm(range(len(args.denoisors)), leave=False)

        for idx in pbar:
            # print info
            model = Denoisor(idx, args)
            info = (model.name.center(10, " "), dataset.name.ljust(15, " "))
            pbar.set_description("Implementing %s to inference on  / %s" % info)
            
            pool = multiprocessing.Pool(processes=args.process)
            for result in pool.starmap(Func, zip(dataset.seqs().__iter__())):
                seq, score = result[0], result[1]
                table.update(seq, model, score)

        table.show(mode="summary")

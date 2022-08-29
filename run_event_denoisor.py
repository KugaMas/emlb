import os
import time
import argparse
import numpy as np

from options.options import set_inference_options, Database
from scripts.denoisors import Denoisor
from scripts.utils.file_utils import load_file, save_file, search_file
from scripts.utils.event_utils import calc_event_structural_ratio
from tqdm import tqdm

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Deployment of EMLB benchmark')
    parser.add_argument('-i', '--input_path', type=str, default='/media/kuga/瓜果山/Datasets/tmp', help='path to load dataset')
    parser.add_argument('-o', '--output_path', type=str, default='results', help='path to output dataset')
    parser.add_argument('-d', '--denoisors', type=list, default=['raw'], help='choose denoisors')
    parser.add_argument("-p", "--params", type=float, default=[], nargs='+', help="specified parameters")

    parser.add_argument('-w', '--save_file', action='store_false', help="save denoising result")
    parser.add_argument('-s', '--calc_esr_score', action='store_false', help="ecaluate esr performance")

    args = set_inference_options(parser)
    
    for dataset in Database(args):
        table = dataset.get_table()
        
        for idx in range(len(args.denoisors)):
            model, fileSeq = Denoisor(idx, args), dataset.iter()

            pbar = tqdm(fileSeq)
            for file in pbar:
                info = (model.name, fileSeq.subname.split('/')[1])
                pbar.set_description("Now implementing %10s to inference on %15s" % info)

                output_path, search_flag, replace_flag = search_file(args, model, fileSeq)
                # skip existing files
                if search_flag and not replace_flag:
                    if not args.calc_esr_score: continue
                    # load denoised event data
                    ev, fr, size = load_file(output_path, aps=fileSeq.use_aps)
                else:
                    # load noisy event data and perform inference
                    ev, fr, size = load_file(fileSeq.path, aps=fileSeq.use_aps)
                    ev = model.run(ev, fr, size)
                    # save inference result
                    if args.save_file:
                        save_file(ev, fr, size, model.params, output_path)

                # calculate ESR
                score = calc_event_structural_ratio(ev, size)            
                table.update(fileSeq, model, score)

        table.show()

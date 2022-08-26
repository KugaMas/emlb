import os
import time
import argparse
import numpy as np

from options.options import set_inference_options, Database
from scripts.denoisors import Denoisor
from scripts.utils.file_utils import load_file, save_file
from scripts.utils.event_utils import calc_event_structural_ratio
from tqdm import tqdm

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Deployment of EMLB benchmark')
    parser.add_argument('-i', '--input_path', type=str, default='/media/kuga/瓜果山/Datasets/DND21', help='path to load dataset')
    parser.add_argument('-o', '--output_path', type=str, default='results', help='path to output dataset')
    parser.add_argument('-d', '--denoisors', type=list, default=['raw', 'baf', 'nn', 'knoise', 'dwf', 'evflow', 'ynoise', 'timesurface', 'fsae', 'iets', 'mlpf', 'edncnn'], help='choose denoisors')
    parser.add_argument("-p", "--params", type=float, default=[], nargs='+', help="specified parameters")

    parser.add_argument('-w', '--save_file', action='store_false', help="save denoising result")
    parser.add_argument('-s', '--calc_esr_score', action='store_false', help="ecaluate esr performance")

    args = set_inference_options(parser)
    
    for dataset in Database(args):
        table = dataset.get_table()
        
        for idx in range(len(args.denoisors)):
            model, fdata = Denoisor(idx, args), dataset.iter()

            pbar = tqdm(fdata)
            score = []
            for file in pbar:
                info = (model.name, fdata.subname.split('/')[1])
                pbar.set_description("Now implementing %10s to inference on %15s" % info)

                output_path = f"{args.output_path}/{model.name}/{fdata.name}/{fdata.subname}.{args.output_file_type}"

                # skip existing files
                if os.path.exists(output_path) and not args.replace_file:
                    if not args.calc_esr_score: continue
                    # load denoised event data
                    ev, fr, size = load_file(output_path, aps=fdata.use_aps, size=fdata.size)

                else:
                    # load noisy event data and perform inference
                    ev, fr, size = load_file(fdata.path, aps=fdata.use_aps, size=fdata.size)
                    ev = model.run(ev, fr, size)

                    # save inference result
                    if args.save_file:
                        save_file(ev, fr, model.params, output_path)

                # calculate ESR
                score = calc_event_structural_ratio(ev, size)            
                table.update(fdata, model, score)

        table.show()

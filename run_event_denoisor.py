import os
import time
import argparse
import numpy as np

from options.options import set_inference_options, Database
from scripts.denoisors import Denoisor
from scripts.utils.file_utils import load_file, save_file
from tqdm import tqdm

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Deployment of EMLB benchmark')
    parser.add_argument('-i', '--input_path', type=str, default='datasets', help='path to load dataset')
    parser.add_argument('-o', '--output_path', type=str, default='results', help='path to output dataset')
    parser.add_argument('-d', '--denoisors', type=list, default=['dwf','baf', 'mlpf'], help='choose denoisors')
    parser.add_argument("-p", "--params", type=float, default=[[],[],[]], nargs='+', help="specified parameters")
    args = set_inference_options(parser)
    
    for idx in range(len(args.denoisors)):
        model = Denoisor(idx, args) # load model
        for fdata in Database(args):
            pbar = tqdm(fdata)
            for file in pbar:
                output_path = f"{args.output_path}/{model.name}/{fdata.name}/{fdata.subname}.{args.output_file_type}"
                # print progress
                info = (model.name, fdata.subname.split('/')[1])
                pbar.set_description("Now implementing %10s to inference on %15s" % info)

                # skip existing files
                # if os.path.exists(output_path): continue

                # load event data and perform inference
                ev, fr, size = load_file(fdata.path, aps=fdata.use_aps, size=fdata.size)
                
                # start model inference
                ev = model.run(ev, fr, size)

                # save inference result
                save_file(ev, fr, model, output_path)

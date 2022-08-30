from tqdm import tqdm
import os
import glob
import yaml
import pandas as pd
import os.path as osp
from tabulate import tabulate

from distutils.command.config import config


class Table:
    def __init__(self, _name, _index):
        self.name = _name
        self.data = pd.DataFrame(index=[osp.basename(osp.splitext(f)[0]) for f in _index])

    def update(self, file, model, score):
        _index = file.name
        _column = model.name
        self.data.loc[_index, _column] = score

    def show(self, mode="summary"):
        self.data.loc['MESR'], _headers = self.data.mean(axis=0), self.data.columns.to_list()

        if mode == "details":
            _headers.insert(0, "files")
            print(tabulate(self.data.iloc[:],   headers=_headers, tablefmt="grid", floatfmt=(".3f")))
        elif mode == "summary":
            _headers.insert(0, self.name)
            print(tabulate(self.data.iloc[-1:], headers=_headers, tablefmt="grid", floatfmt=(".3f")))


class fileSeq:
    def __init__(self, path, use_aps):
        self.path = path
        self.use_aps = use_aps

        fpath, fname  = osp.split(self.path)
        fname, fext   = osp.splitext(fname)
        self.name     = fname
        self.subname  = f"{osp.basename(fpath)}/{fname}"


class Dataset:
    def __init__(self, name, path, fclass, use_aps):
        self.name    = name
        self.path    = path
        self.fclass  = fclass
        self.use_aps = use_aps

        # load all file paths
        self.file_paths = [file for folder in fclass for file in glob.glob(f"{path}/{folder}/*")]
        self.file_nums  = len(self.file_paths)
        self.file_paths.sort()
    
    def seqs(self):
        args = (self.use_aps,)
        return [fileSeq(path, *args) for path in self.file_paths]

    def table(self):
        return Table(self.name, self.file_paths)


class Database:
    def __init__(self, args, save_path='options/dataset_info.yaml'):
        self.path, self.loader = args.input_path, dict()
        for fname in os.listdir(self.path):
            if fname[0] == '.': continue
            self.loader[fname] = os.listdir(f"{self.path}/{fname}")
        self.iterator = iter(self.loader.keys())

        # read config data
        if not osp.exists(save_path):
            self.config = dict()
        else:
            with open(save_path) as f:
                self.config = yaml.load(f, Loader=yaml.FullLoader)

        # add information
        for fname, flist in self.loader.items():
            if fname in self.config.keys(): continue
            self.config[fname] = dict()
            self.config[fname]['frame'] = False
            self.config[fname]['class'] = flist

        # write config data
        with open(save_path, "w") as f:
            yaml.dump(self.config, f, encoding='utf-8', allow_unicode=True)

    def __iter__(self):
        return self

    def __next__(self):
        _fname = next(self.iterator)
        _fpath = f"{self.path}/{_fname}"

        _frame = self.config[_fname]['frame']
        _class = self.config[_fname]['class']
        
        return Dataset(_fname, _fpath, _class, _frame)


def set_inference_options(parser):
    """ Fundamental Information Settings """
    parser.add_argument('--replace_file', action='store_true', help="replace the former output file")
    parser.add_argument('--output_file_type', type=str, default='pkl', help='output file type')

    """ Data Preprocess Settings """
    parser.add_argument('--use_polarity',  type=bool, default=True)
    parser.add_argument('--excl_hotpixel', type=bool, default=True)

    """ Parser """
    args = parser.parse_args()
    # assert len(args.denoisors) == len(args.params), "The number of denoisors must match parameters"

    """ Load Parameters Preparation """
    args.abs_path    = os.getcwd()
    args.input_path  = osp.join(args.abs_path, args.input_path)
    args.output_path = osp.join(args.abs_path, args.output_path)
    args.denoisors   = [denoisor.lower() for denoisor in args.denoisors]

    """ Iteratable Dataset Reader """
    args.database = Database(args)

    return args

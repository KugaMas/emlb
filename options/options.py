from tqdm import tqdm
import os
import glob
import yaml
import pandas as pd
import os.path as osp
from tabulate import tabulate

from distutils.command.config import config


class Dataset_iter:
    def __init__(self, name, path, size, fclass, use_aps):
        self.name    = name
        self.size    = size
        self.fclass  = fclass
        self.use_aps = use_aps
        
        # load all file paths
        self.file_paths = [file for folder in fclass for file in glob.glob(f"{path}/{folder}/*")]
        self.file_nums  = len(self.file_paths)
        self.file_paths.sort()
        self.file_paths = self.file_paths.__iter__()

        # initialize nullptr parameters
        self.ev      = None
        self.fr      = None
        self.path    = None
        self.subname = None

    def __iter__(self):
        return self

    def __next__(self):
        self.path  = self.file_paths.__next__()
        path, name = osp.split(self.path)
        name, ext  = osp.splitext(name)
        self.subname = f"{osp.basename(path)}/{name}"
        return self.path

    def __len__(self):
        return self.file_nums


class Table:
    def __init__(self, _index):
        self.data = pd.DataFrame(index=[osp.basename(osp.splitext(f)[0]) for f in _index])

    def update(self, file, model, score):
        _index = osp.basename(file.subname)
        _column = model.name
        self.data.loc[_index, _column] = score

    def show(self):
        self.data.loc['MESR'] = self.data.mean(axis=0)

        _headers = self.data.columns.to_list()
        _headers.insert(0, "files")
        print(tabulate(self.data.iloc[:-1], headers=_headers, tablefmt="grid", floatfmt=(".2f")))

        _headers[0] = ' '
        print(tabulate(self.data.iloc[-1:], headers=_headers, tablefmt="grid", floatfmt=(".2f")))


class Dataset:
    def __init__(self, name, path, size, fclass, use_aps):
        self.name    = name
        self.path    = path
        self.size    = size
        self.fclass  = fclass
        self.use_aps = use_aps

    def iter(self):
        return Dataset_iter(self.name, self.path, self.size, self.fclass, self.use_aps)

    def get_table(self):
        return Table(self.iter())


class Database:
    def __init__(self, args, save_path='options/dataset_info.yaml'):
        self.path, self.loader = args.input_path, dict()
        for fname in os.listdir(self.path):
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
            self.config[fname]['size']  = (346, 260)
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

        _size  = self.config[_fname]['size']
        _frame = self.config[_fname]['frame']
        _class = self.config[_fname]['class']
        
        return Dataset(_fname, _fpath, _size, _class, _frame)


def set_inference_options(parser):
    """ Fundamental Information Settings """
    parser.add_argument('--replace_file', action='store_true', help="replace the former output file")
    parser.add_argument('--output_file_type', type=str, default='txt', help='output file type')

    """ Data Preprocess Settings """
    parser.add_argument('--use_polarity',  type=bool, default=True)
    parser.add_argument('--excl_hotpixel', type=bool, default=True)

    """ Parser """
    args = parser.parse_args()
    assert len(args.denoisors) == len(args.params), "The number of denoisors must match parameters"

    """ Load Parameters Preparation """
    args.abs_path    = os.getcwd()
    args.input_path  = osp.join(args.abs_path, args.input_path)
    args.output_path = osp.join(args.abs_path, args.output_path)

    """ Iteratable Dataset Reader """
    args.database = Database(args)

    return args

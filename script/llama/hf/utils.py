import glob
from logging import getLogger, Formatter, StreamHandler, DEBUG, INFO
import os

import torch

def _set_handler(logger, handler, verbose: bool):
    """
    Prep handler
    """
    if verbose:
        handler.setLevel(DEBUG)
    else:
        handler.setLevel(INFO)
    formatter = Formatter(
        "%(asctime)s %(name)s:%(lineno)s [%(levelname)s]: %(message)s"
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    return logger


def get_module_logger(verbose: bool = False, level=DEBUG):
    """
    Create logger
    """
    logger = getLogger(__name__)
    logger = _set_handler(logger, StreamHandler(), False)
    logger.setLevel(level)
    logger.propagate = False
    return logger


def makedirs_recursive(path):
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)


def read_result(path: str, file_type: str="result", extract_nums: list=[], extension: str="txt",):
    out = []
    if not len(extract_nums):
        files = glob.glob(f"{path}/{file_type}*.{extension}")
    else:
        files = []
        for n in extract_nums:
            fl = f"{path}/{file_type}{n}.{extension}"
            files.append(fl)
    out_nums = []
    for fl in sorted(files):
        out_num = fl.split(file_type)[-1].split(f".{extension}")[0]
        if extension == "txt":
            with open(fl, "r") as f:
                result = f.read()
        elif extension == "pt":
            result = torch.load(fl)
        else:
            raise NotImplementedError(f"Extension {extension} not implemented")
        out.append(result)
        out_nums.append(out_num)
    return out, out_nums


def list_iid_dirs(path):
    return os.listdir(path)


def load_specified_files(path: str, file_type: str="result", extract_nums=[], extension: str="txt"):
    out, out_nums = {}, {}
    for iid in list_iid_dirs(path):
        if "." in iid:
            continue 
        ns = extract_nums[iid] if type(extract_nums)==dict else extract_nums
        out[iid], out_nums[iid] = read_result(f"{path}/{iid}", file_type=file_type, extract_nums=ns, extension=extension,)
    return out, out_nums
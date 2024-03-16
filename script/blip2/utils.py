from logging import getLogger, Formatter, StreamHandler, DEBUG, INFO

from dask import delayed
import dask.dataframe as dd
import pandas as pd


@delayed
def read_jsonl_w_source(file_path, split):
    """
    Read single jsonl as dataframe w/ split info
    """
    df = pd.read_json(file_path, orient="records", lines=True).rename(
        {"image": "img"}, axis=1
    )
    df["split"] = split
    return df.drop_duplicates()


def read_jsonls_w_source(
    file_path,
    splits=["dev_seen", "dev_unseen", "train", "test_seen", "test_unseen"],
):
    """
    Concatenate jsonl files to a dataframe w/ split info
    """
    ddf_memes = dd.from_delayed(
        [read_jsonl_w_source(f"{file_path}/{s}.jsonl", s) for s in splits]
    )
    df_memes = ddf_memes.compute()
    return df_memes


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


def activate_attn(attn, device):
    attn = attn.to(device)
    attn.requires_grad_(True)
    return attn

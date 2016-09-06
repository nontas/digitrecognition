import os
from pathlib import Path


def src_dir_path():
    r"""
    The path to the top of the package.

    Returns
    -------
    path : `pathlib.Path`
        The full path to the top of the package.
    """
    return Path('/home/nontas/Documents/Research/digitrecognition/')
    #return Path(os.path.abspath(__file__)).parent.parent

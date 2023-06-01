import os
import yaml

__all__ = ["file_path", "load_yaml"]


def file_path(string):
    if os.path.isfile(string):
        return string
    else:
        raise NotADirectoryError(string)


def load_yaml(path):
    with open(path) as file:
        res = yaml.load(file, Loader=yaml.FullLoader)
    return res

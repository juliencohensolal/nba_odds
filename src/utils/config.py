import shutil
import yaml


class Config(object):
    def __init__(self, adict):
        self.__dict__.update(adict)
        for key, value in adict.items():
            if isinstance(value, dict):
                self.__dict__[key] = Config(value)


def load_config(file):
    with open(file, "rb") as stream:
        dict_config = yaml.load(stream, Loader=yaml.FullLoader)
        return Config(dict_config)


def save_config(dir_in, file_in, dir_out, suffix=""):
    shutil.copy2(dir_in + "/" + file_in, dir_out + "/" + file_in.split(".yml")[0] + suffix + ".yml")

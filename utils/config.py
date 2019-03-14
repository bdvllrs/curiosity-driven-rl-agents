__author__ = "Benjamin Devillers (bdvllrs)"
__version__ = "1.0.5"

import os
import yaml
from utils.singleton import Singleton

__all__ = ["config"]


def update_config(conf, new_conf):
    for item in new_conf.keys():
        if type(new_conf[item]) == dict and item in conf.keys():
            conf[item] = update_config(conf[item], new_conf[item])
        else:
            conf[item] = new_conf[item]
    return conf


class Config:
    def __init__(self, path="config/", cfg=None):
        self.__is_none = False
        self.__data = cfg if cfg is not None else {}
        if path is not None and cfg is None:
            self.__path = os.path.abspath(os.path.join(os.curdir, path))
            with open(os.path.join(self.__path, "default.yaml"), "rb") as default_config:
                self.__data.update(yaml.load(default_config, Loader=yaml.FullLoader))
            for cfg in sorted(os.listdir(self.__path)):
                if cfg != "default.yaml" and cfg[-4:] in ["yaml", "yml"]:
                    with open(os.path.join(self.__path, cfg), "rb") as config_file:
                        self.__data = update_config(self.__data, yaml.load(config_file, Loader=yaml.FullLoader))

    def set_(self, key, value):
        self.__data[key] = value

    def values_(self):
        return self.__data

    def save_(self, file):
        file = os.path.abspath(os.path.join(os.curdir, file))
        with open(file, 'w') as f:
            yaml.dump(self.__data, f)

    def __getattr__(self, item):
        if type(self.__data[item]) == dict:
            return Config(cfg=self.__data[item])
        return self.__data[item]

    def __getitem__(self, item):
        return self.__data[item]


config = Singleton(Config)

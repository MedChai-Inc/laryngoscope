import yaml
import PIL
import numpy as np

def load_yaml(file_name):
    with open(file_name, 'r') as stream:
        config = yaml.load(stream, Loader=yaml.SafeLoader)
    return config

import torch
import numpy
import torchvision.datasets as dataset
import torchvision.transforms as transforms

from helperFunc import load_yaml

trainingSet = dataset.CocoCaptions('/Users/carsonstillman/Documents/EPAD_Honors/U-Net/data/EVMS1/EMS\ Real\ 1/','/Users/carsonstillman/Documents/EPAD_Honors/U-Net/data/EVMS1/EVMS_1_export_2022-03-07_17-32-16.json',)


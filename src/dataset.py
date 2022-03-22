import glob
import os
from PIL import Image

from torch.utils import data
from torchvision.datasets.folder import pil_loader

import json


def load_json(f):
    with open(f, 'r') as fp:
        return json.load(fp)


class BDDDataset(data.Dataset):

    def __init__(self, root, train=True, transform=None):
        self.root = root
        self.train = train
        self.transform = transform
        self.samples = None

        self.prepare()

    def prepare(self):
        self.samples = []

        if self.train:
            label_path = glob.glob(
                os.path.join(self.root, '*.json'))
            image_dir = os.path.join(self.root, '')
            image_paths = glob.glob(
                os.path.join(self.root, '*.jpg'))
        else:
            label_path = glob.glob(
                os.path.join(self.root, '*.json'))
            image_dir = os.path.join(self.root, '')
            image_paths = glob.glob(
                os.path.join(self.root, '*.jpg'))

        for image_path in image_paths:

            if os.path.exists(image_path):
                self.samples.append((image_path, label_path[0]))
            else:
                raise FileNotFoundError

    def __getitem__(self, index):
        # TODO: handle label dict

        image_path, label_path = self.samples[index]

        image = pil_loader(image_path)
        loaded_json = load_json(label_path)
        #something is wrong here
        for frame in loaded_json['frames']:
            name = frame['name']
            if name[name.rfind('/')+1] == image_path[image_path.rfind('/')+1]:
                label = frame
                
        if self.transform is not None:
            image = self.transform(image)

        return image, label

    def __len__(self):
        return len(self.samples)


def main():
    from torchvision import transforms
    transform = transforms.Compose(
        [transforms.Resize(64), transforms.ToTensor()])
    loader = data.DataLoader(
        BDDDataset(r'data/EVMS1/EMS Real 1/', transform=transform),
        batch_size=1,
        shuffle=True)

    count = 0
    for i, (x, y) in enumerate(loader):
        count+=1
        print(x.size())
        print(y)
        if count>1:
            break


if __name__ == '__main__':
    main()
import glob
import os
from PIL import Image

from torch.utils import data
from torchvision.datasets.folder import pil_loader

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
                os.path.join(self.root, 'training_masks/*.jpg'))
            image_dir = os.path.join(self.root, '')
            image_paths = glob.glob(
                os.path.join(self.root, 'training_images/*.jpg'))
        else:
            label_path = glob.glob(
                os.path.join(self.root, 'testing_masks/*.jpg'))
            image_dir = os.path.join(self.root, '')
            image_paths = glob.glob(
                os.path.join(self.root, 'testing_images/*.jpg'))

        for image_path in image_paths:

            if os.path.exists(image_path):
                self.samples.append((image_path, label_path[0]))
            else:
                raise FileNotFoundError

    def __getitem__(self, index):
        # TODO: handle label dict

        image_path, label_path = self.samples[index]

        image = pil_loader(image_path)
        
                
        if self.transform is not None:
            image = self.transform(image)

        return image, label

    def __len__(self):
        return len(self.samples)


def main():
    from torchvision import transforms
    transform = transforms.Compose(
        [transforms.Resize(64), transforms.ToTensor()])
    #must supply your path to the data
    loader = data.DataLoader(
        BDDDataset(r'/Users/carsonstillman/data/Epiglottis_Data/', transform=transform),
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
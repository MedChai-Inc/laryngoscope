import torch
import numpy
import dataset
from torchvision import transforms
from torch.utils.data import DataLoader

#compose the all transformation
transform = transforms.Compose([transforms.Resize(64), transforms.ToTensor()])

#define the training dataset and the testing dataset
training_data = dataset.ETIDAtaset(
    r'/Users/carsonstillman/data/Epiglottis_Data/', train=True, transform=transform)

test_data = dataset.ETIDataset(
    r'/Users/carsonstillman/data/Epiglottis_Data/', train=False, transform=transform)

#instantiate the dataloaders
train_dataloader = DataLoader(training_data, batch_size=64)
test_dataloader = DataLoader(test_data, batch_size=64)

def train_loop():
    '''Runs the training loop for the Neural network.'''
    pass

def test_loop():
    '''Runs the testing loop for the Neural Network.'''
    pass

def main():
    '''The main function'''

if __name__=='__main__':
    main()


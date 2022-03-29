import torch
import numpy
import dataset
from model import UNet
from torch import nn
from torchvision import transforms
from torch.utils.data import DataLoader

#compose the all transformation
transform = transforms.Compose([transforms.Resize(64), transforms.ToTensor()])

#define the training dataset and the testing dataset
training_data = dataset.ETIDAtaset(
    r'/Users/carsonstillman/data/Epiglottis_Data/', train=True, transform=transform)

test_data = dataset.ETIDataset(
    r'/Users/carsonstillman/data/Epiglottis_Data/', train=False, transform=transform)

def train_loop():
    '''Runs the training loop for the Neural network.'''
    pass

def test_loop():
    '''Runs the testing loop for the Neural Network.'''
    pass

def main():
    '''The main function'''
    #instantiate a UNet model object
    model = UNet()

    #define variables for ML
    learning_rate = 1e-3
    batch_size = 100
    epochs = 5

    #instantiate the dataloaders
    train_dataloader = DataLoader(training_data, batch_size=batch_size)
    test_dataloader = DataLoader(test_data, batch_size=batch_size)  

    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

    for t in range(epochs):
        print(f"Epoch {t+1}\n-------------------------------")
        train_loop(train_dataloader, model, loss_fn, optimizer)
        test_loop(test_dataloader, model, loss_fn)
    print("Done!")

    PATH = './object_detection.pth'
    torch.save(model.state_dict(), PATH)

if __name__=='__main__':
    main()


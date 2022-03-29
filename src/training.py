import torch
import numpy
import dataset
from model import UNet
from torch import nn
from torchvision import transforms
from torch.utils.data import DataLoader

def train_loop(dataloader, model, loss_fn, optimizer):
    '''Runs the training loop for the Neural network.'''
    size = len(dataloader.dataset)
    for batch, (X, y) in enumerate(dataloader):
        # Compute prediction and loss
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 100 == 0:
            loss, current = loss.item(), batch * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

def test_loop(dataloader, model, loss_fn):
    '''Runs the testing loop for the Neural Network.'''
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss, correct = 0, 0

    with torch.no_grad():
        for X, y in dataloader:
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()

    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")

def main():
    '''The main function'''
    #instantiate a UNet model object
    model = UNet()

    #define variables for ML
    learning_rate = 1e-3
    batch_size = 100
    epochs = 5

    #compose the all transformation
    transform = transforms.ToTensor()

    #define the training dataset and the testing dataset
    training_data = dataset.ETIDataset(
    r'/Users/carsonstillman/data/Epiglottis_Data/', train=True, transform=transform)

    test_data = dataset.ETIDataset(
    r'/Users/carsonstillman/data/Epiglottis_Data/', train=False, transform=transform)

    #instantiate the dataloaders
    train_dataloader = DataLoader(training_data, batch_size=batch_size)
    test_dataloader = DataLoader(test_data, batch_size=batch_size)  

    #define a loss function
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
    optimizer.zero_grad()

    for t in range(epochs):
        print(f"Epoch {t+1}\n-------------------------------")
        train_loop(train_dataloader, model, loss_fn, optimizer)
        test_loop(test_dataloader, model, loss_fn)
    print("Done!")

    #PATH = './object_detection.pth'
    #torch.save(model.state_dict(), PATH)

if __name__=='__main__':
    main()


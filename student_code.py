# python imports
import os
from tqdm import tqdm

# torch imports
import torch
import torch.nn as nn
import torch.optim as optim

# helper functions for computer vision
import torchvision
import torchvision.transforms as transforms


class LeNet(nn.Module):
    def __init__(self, input_shape=(32, 32), num_classes=100):
        super(LeNet, self).__init__()
        # certain definitions
        self.input_shape = input_shape

        # One convolutional layer with the number of output channels to be 6, kernel size to be 5, stride to be 1,
        # followed by a relu activation layer and then a 2D max pooling layer
        self.conv1 = nn.Conv2d(3, 6, 5, stride=1)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(2, stride=2)
        # One convolutional layer with the number of output channels to be 16, kernel size to be 5, stride to be 1,
        # followed by a relu activation layer and then a 2D max pooling layer
        self.conv2 = nn.Conv2d(6, 16, 5, stride=1)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(2, stride=2)
        # A Flatten layer to convert the 3D tensor to a 1D tensor
        self.flatten = nn.Flatten()
        # A Linear layer with output dimension to be 256, followed by a relu activation function.
        self.linear1 = nn.Linear((16 * 5 * 5), 256)
        self.relu3 = nn.ReLU()
        # A Linear layer with output dimension to be 128, followed by a relu activation function.
        self.linear2 = nn.Linear(256, 128)
        self.relu4 = nn.ReLU()
        # A Linear layer with output dimension to be the number of classes (in our case, 100)
        self.output = nn.Linear(128, 100)


    def forward(self, x):
        # certain operations
        # follow steps 1 - 6 based on the specification
        # convolve
        x = self.conv1(x)
        # perofrm ReLu
        x = self.relu1(x)
        # pool
        x = self.pool1(x)
        step1 = x.size()
        # convolve again
        x = self.conv2(x)
        # perform ReLu again
        x = self.relu2(x)
        # pool
        x = self.pool2(x)
        step2 = x.size()

        x = self.flatten(x)
        step3 = x.size()

        x = self.linear1(x)
        x = self.relu3(x)
        step4 = x.size()

        x = self.linear2(x)
        x = self.relu4(x)
        step5 = x.size()

        x = self.output(x)
        step6 = x.size()

        shape_dict = {1: list(step1), 2: list(step2), 3: list(step3), 4: list(step4), 5: list(step5), 6: list(step6)}
        out = x
        return out, shape_dict


def count_model_params():
    '''
    return the number of trainable parameters of LeNet.
    '''
    model = LeNet()
    # model_params = list(len(model.named_parameters()))
    # model_params = list(model.named_parameters()).__len__()
    # model_params = len(list(model.named_parameters()))

    model_params = 0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad: continue
        params = parameter.numel()
        model_params += params

    return model_params


def train_model(model, train_loader, optimizer, criterion, epoch):
    """
    model (torch.nn.module): The model created to train
    train_loader (pytorch data loader): Training data loader
    optimizer (optimizer.*): A instance of some sort of optimizer, usually SGD
    criterion (nn.CrossEntropyLoss) : Loss function used to train the network
    epoch (int): Current epoch number
    """
    model.train()
    train_loss = 0.0
    for input, target in tqdm(train_loader, total=len(train_loader)):
        ###################################
        # fill in the standard training loop of forward pass,
        # backward pass, loss computation and optimizer step
        ###################################

        # 1) zero the parameter gradients
        optimizer.zero_grad()
        # 2) forward + backward + optimize
        output, _ = model(input)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

        # Update the train_loss variable
        # .item() detaches the node from the computational graph
        # Uncomment the below line after you fill block 1 and 2
        train_loss += loss.item()

    train_loss /= len(train_loader)
    print('[Training set] Epoch: {:d}, Average loss: {:.4f}'.format(epoch+1, train_loss))

    return train_loss


def test_model(model, test_loader, epoch):
    model.eval()
    correct = 0
    with torch.no_grad():
        for input, target in test_loader:
            output, _ = model(input)
            pred = output.max(1, keepdim=True)[1]
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_acc = correct / len(test_loader.dataset)
    print('[Test set] Epoch: {:d}, Accuracy: {:.2f}%\n'.format(
        epoch+1, 100. * test_acc))

    return test_acc

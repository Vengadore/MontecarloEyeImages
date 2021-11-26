import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
import torch
import torchvision
from torchvision.datasets.utils import download_url
from torch.utils.data import random_split
from torchvision.datasets import ImageFolder
from torchvision import transforms
from torchvision.transforms import ToTensor
from torch.utils.data.dataloader import DataLoader
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt


from model import ImageClassificationModel

###################################
## Define the origin of the DATA ##
###################################

base_dir = "./flowers"
print("The following classes were found {}".format(os.listdir(base_dir)))



#########################################################
## Define the transformations to perform to the images ##
#########################################################

transformer = torchvision.transforms.Compose(
    [  # Applying Augmentation
        torchvision.transforms.Resize((224, 224)),
        torchvision.transforms.RandomHorizontalFlip(p=0.5),
        torchvision.transforms.RandomVerticalFlip(p=0.5),
        torchvision.transforms.RandomRotation(30),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        ),
    ]
)
# Load the dataset and apply the transformations
dataset = ImageFolder(base_dir, transform=transformer)

# Set the length for training and validation
validation_size = int(0.4*len(dataset))
training_size = len(dataset) - validation_size

# Split the classes into validation and training
train_ds, val_ds_main = random_split(dataset,[training_size, validation_size])
# Split validation into Validation and Test
val_ds, test_ds  = random_split(val_ds_main,[int(validation_size*0.6), len(val_ds_main)-int(validation_size*0.6)])

print("The number of data points for training is {}".format(len(train_ds)))

# Create the DataLoaders for the images
train_dl = DataLoader(train_ds, batch_size = 32, shuffle=True)
val_dl = DataLoader(val_ds, batch_size = 32)
test_dl = DataLoader(test_ds, batch_size = 32)

model = ImageClassificationModel()
print(model)


# Define the evaluation routine (No gradient)
@torch.no_grad()
def evaluate(model, val_loader):
    model.eval()
    outputs = [model.validation_step(batch) for batch in val_loader]
    return model.validation_epoch_end(outputs)
# Fit function
def fit(epochs, lr, model, train_loader, val_loader, opt_func=torch.optim.Adam):
    history = []
    optimizer = opt_func(model.parameters(), lr)
    for epoch in range(epochs):
        # Training Phase 
        model.train()
        train_losses = []
        for batch in train_loader:
            loss = model.training_step(batch)
            train_losses.append(loss)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        # Validation phase
        result = evaluate(model, val_loader)
        result['train_loss'] = torch.stack(train_losses).mean().item()
        model.epoch_end(epoch, result)
        history.append(result)
    return history


def get_default_device():
    """Pick GPU if available, else CPU"""
    if torch.cuda.is_available():
        return torch.device('cuda')
    else:
        return torch.device('cpu')
    
def to_device(data, device):
    """Move tensor(s) to chosen device"""
    if isinstance(data, (list,tuple)):
        return [to_device(x, device) for x in data]
    return data.to(device, non_blocking=True)

class DeviceDataLoader():
    """Wrap a dataloader to move data to a device"""
    def __init__(self, dl, device):
        self.dl = dl
        self.device = device
        
    def __iter__(self):
        """Yield a batch of data after moving it to device"""
        for b in self.dl: 
            yield to_device(b, self.device)

    def __len__(self):
        """Number of batches"""
        return len(self.dl)
device = get_default_device()


train_dl = DeviceDataLoader(train_dl, device)
val_dl = DeviceDataLoader(val_dl, device)
to_device(model, device);


model = to_device(ImageClassificationModel(), device)
evaluate(model, val_dl)

# Training parameters
num_epochs = 40
opt_func = torch.optim.Adam
lr = 0.001

###################
## TRAIN NETWORK ##
###################
history = fit(num_epochs, lr, model, train_dl, val_dl, opt_func)


test_dl = DeviceDataLoader(test_dl, device)
evaluate(model, test_dl)

torch.save(model,"./trained_model/final_model.ph")
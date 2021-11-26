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
    [  # Applying common transformations to the image
        torchvision.transforms.Resize((224, 224)),
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
test_dl = DataLoader(test_ds, batch_size = 1)

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
device = get_default_device()


model = ImageClassificationModel()
model = torch.load("./trained_model/final_model.ph")

samples = 20
with torch.no_grad():
    results = []
    a = [f for f in test_dl]
    image = to_device(a[0][0],device)
    target = to_device(a[0][1],device)
    
    for i in range(samples):
        results.append(model(image))

print(results)
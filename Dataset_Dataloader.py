import torch
import torchvision
from torch.utils.data import Dataset, DataLoader
import numpy as np
import math

class WineDataset(Dataset):

    def __init__(self):
        # data loading
        xy = np.loadtxt('/Users/wantinghe/Documents/02 PyTorch/Data/wine.csv',
                        delimiter=",", dtype=np.float32, skiprows=1)
        self.x = torch.from_numpy(xy[:,1:])
        self.y = torch.from_numpy(xy[:,[0]])  #n_samples
        self.n_samples = xy.shape[0]

    def __getitem__(self, index):
        # for indexing， dataset[0]
        return self.x[index], self.y[index]

    def __len__(self):
        # len function， len(dataset)
        return self.n_samples

# create dataset
dataset = WineDataset()

# get first sample and unpack
first_data = dataset[0]
features, labels = first_data
print(features, labels)

# Load whole dataset with DataLoader
# shuffle: shuffle data, good for training
# num_workers: faster loading with multiple subprocesses
# !!! IF YOU GET AN ERROR DURING LOADING, SET num_workers TO 0 !!!
dataloader = DataLoader(dataset=dataset, batch_size=4, shuffle=True, num_workers=0)

# convert to an iterator and look at one random sample
dataiter = iter(dataloader)
data = next(dataiter)
features, labels = data
print(features, labels)

# training loop
num_epochs = 2
total_samples = len(dataset)
n_iterations = math.ceil(total_samples/4)
print(total_samples, n_iterations)  # output 178, 45

for epoch in range(num_epochs):
    for i, (inputs, labels) in enumerate(dataloader):
        # forward, backward, update
        # here: 178 samples, batch_size = 4, n_iters=178/4=44.5 -> 45 iterations
        if (i+1) %5 == 0:
            print(f'epoch {epoch+1}/{num_epochs}, step {i+1}/{n_iterations}, input {inputs.shape}')

# some famous datasets are available in torchvision.datasets
# e.g. MNIST, Fashion-MNIST, CIFAR10, COCO

train_dataset = torchvision.datasets.MNIST(root='./data',
                                           train=True,
                                           transform=torchvision.transforms.ToTensor(),
                                           download=True)

train_loader = DataLoader(dataset=train_dataset,
                                           batch_size=3,
                                           shuffle=True)

# look at one random sample
dataiter = iter(train_loader)
data = next(dataiter)
inputs, targets = data
print(inputs.shape, targets.shape)

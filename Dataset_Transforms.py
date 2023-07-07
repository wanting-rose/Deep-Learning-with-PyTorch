import torch
import torchvision
from torch.utils.data import Dataset
import numpy as np

# PyTorch has built-in transforms
# e.g. ToTensor: convert images or numpy array to tensors
# dataset = torchvision.datasets.MNIST(
#     root='./data', transform=torchvision.transforms.ToTensor())

class WineDataset(Dataset):

  def __init__(self, transform=None):
    # data loading
    xy = np.loadtxt('/Users/wantinghe/Documents/02 PyTorch/Data/wine.csv',
                    delimiter=",", dtype=np.float32, skiprows=1)
    self.n_samples = xy.shape[0]

    # Note: no need to convert tensor here
    self.x = xy[:, 1:]
    self.y = xy[:, [0]]  # n_samples

    self.transform = transform

  def __getitem__(self, index):
      # for indexing， dataset[0]
      sample = self.x[index], self.y[index]

      if self.transform:
        sample = self.transform(sample)

      return sample

  def __len__(self):
    # len function， len(dataset)
    return self.n_samples


# Custom Transforms
# 1）ToTensot
class ToTensor():
    def __call__(self, sample):
        inputs, targets = sample
        return torch.from_numpy(inputs), torch.from_numpy(targets)

# <class 'torch.Tensor'> <class 'torch.Tensor'>
dataset = WineDataset(transform=ToTensor())
first_data = dataset[0]
features, labels = first_data
print(type(features), type(labels))

# <class 'numpy.ndarray'> <class 'numpy.ndarray'>
dataset = WineDataset(transform=None)
first_data = dataset[0]
features, labels = first_data
print(type(features), type(labels))

# 2） Multiplication Transform
class MulTransform:
     # multiply inputs with a given factor
    def __init__(self, factor):
        self.factor = factor

    def __call__(self, sample):
        inputs, targets = sample
        inputs *= self.factor
        return inputs, targets

print('Without Transform')
dataset = WineDataset()
first_data = dataset[0]
features, labels = first_data
print(type(features), type(labels))
print(features, labels)

print('\nWith Tensor Transform')
dataset = WineDataset(transform=ToTensor())
first_data = dataset[0]
features, labels = first_data
print(type(features), type(labels))
print(features, labels)

print('\nWith Tensor and Multiplication Transform')
composed = torchvision.transforms.Compose([ToTensor(), MulTransform(4)])
dataset = WineDataset(transform=composed)
first_data = dataset[0]
features, labels = first_data
print(type(features), type(labels))
print(features, labels)

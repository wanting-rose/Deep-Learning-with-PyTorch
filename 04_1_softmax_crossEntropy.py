import torch
import torch.nn as nn
import numpy as np

# Softmax
# 1) calculated with numpy
def softmax(x):
  return np.exp(x) / np.sum(np.exp(x), axis=0)

x = np.array([2.0, 1.0, 0.1])
outputs = softmax(x)
print('softmax numpy:', outputs)

# 2) calculate with torch
x = torch.tensor([2.0, 1.0, 0.1])
outputs = torch.softmax(x, dim=0)
print(outputs)

# Cross-entropy
# Cross-entropy loss, or log loss, measures the performance of a classification model
# whose output is a probability value between 0 and 1.
# Y-One hot encoded class labels, Y_predicted-Probabilities (Softmax)
# -> loss increases as the predicted probability diverges from the actual label

# 1) Calculate with numpy
def cross_entropy(actual, predicted):
    loss = -np.sum(actual * np.log(predicted))
    return loss

Y = np.array([1,0,0])
Y_pred_good = np.array([0.7, 0.2, 0.1])
Y_pred_bad = np.array([0.1, 0.3, 0.6])
l1 = cross_entropy(Y, Y_pred_good)
l2 = cross_entropy(Y, Y_pred_bad)
print(f'loss 1 numpy: {l1:.4f}')
print(f'loss 2 numpy: {l2:.4f}')

# 2) calculate with Torch
# nn.CrossEntropyLoss = nn.LogSoftmax + nn.NLLLoss
# NLLLoss = negative log likelihood loss
loss = nn.CrossEntropyLoss()

Y = torch.tensor([2, 0, 1])
Y_pred_good = torch.tensor(
    [[0.1, 1.0, 2.1],
     [2.0, 1.0, 0.1],
     [0.1, 3.0, 0.1]])
Y_pred_bad = torch.tensor(
    [[2.1, 1.0, 0.1],
     [0.1, 1.0, 2.1],
     [0.1, 3.0, 0.1]])

l1 = loss(Y_pred_good, Y)
l2 = loss(Y_pred_bad, Y)

print(f'PyTorch loss 1: {l1.item():.4f}')
print(f'PyTorch loss 2: {l2.item():.4f}')

_, prediction1 = torch.max(Y_pred_good, 1)
_, prediction2 = torch.max(Y_pred_bad,1)
print(prediction1)
print(prediction2)





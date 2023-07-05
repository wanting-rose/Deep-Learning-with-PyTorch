import numpy as np

# forward pass: compute loss
# compute local gradients
# backward pass: compute dloss/dweights using the chain rule

X = np.array([1,2,3,4], dtype=np.float32)
Y = np.array([2,4,6,8], dtype=np.float32)

w = 0.0

# model prediction
def forward (x):
    return w * x

# loss = MSE
def loss(y, y_predict):
    return ((y_predict-y)**2).mean()

# gradient
def gradient(x,y,y_predict):
    return np.dot(2*x, y_predict-y).mean()

print(f'Prediction before training: f(5)={forward(5):.3f}')

# Training
learning_rate = 0.01
n_iters = 10

for epoch in range(n_iters):
    # prediction = forward pass
    y_pred = forward(X)

    l = loss(Y, y_pred)
    # gradient = backward pass
    dw = gradient(X,Y,y_pred)

    # update weight
    w -= learning_rate * dw

    if epoch % 1 == 0:
        print(f'epoch{epoch+1}: w={w:.3f}, loss={l:.8f}')

print(f'Prediction after training: f(5)={forward(5):.3f}')





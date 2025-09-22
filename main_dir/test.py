import numpy as np

def diff_softmax(y):
    _, l = y.shape
    delta = np.eye(l)[None, :, :] * y[:, :, None] 
    outer = y[:, :, None] * y[:, None, :]
    diff = delta - outer
    return diff

y = np.arange(1, 21)
y = np.reshape(y, (5, 4))
print(y, "\n")

ans = diff_softmax(y)
print(ans.shape, y.shape)
y = np.einsum('nij,ni->nj', ans, y)
print(y.shape)
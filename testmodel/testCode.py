import numpy as np
import copy

v = {} # vocabulary: list for all words which is used in all sentences from datasets.

#----reader----
f = open('zeroToTransformer-/data/dataset.txt', 'r')
dataset = f.read().split()

v = list(set(dataset))

#----ToOnehot----
temp = np.zeros((len(dataset), len(v)))
for i, token in enumerate(dataset):
    temp[i, v.index(token)] = 1

print(temp)

X_train = temp[:-1, :]
y_train =temp[1:, :]

#----model----
rng = np.random.default_rng()

layer = []
layer.append(rng.random((3, 6)))
layer.append(rng.random((6, 12)))
layer.append(rng.random((12, 3)))

def softmax(input):
    maxinput = np.tile(np.reshape(np.max(input), (input.shape[0], 1)), 3)
    temp = np.exp(input)[:, :] - maxinput # 各行の最大値を引いて値を抑え込む
    total = np.reshape(np.sum(temp, axis=1), (input.shape[0], 1))
    total = np.tile(total, 3)
    
    return np.exp(input) / total

def forward(input):
    z1 = input @ layer[0]
    a1 = np.tanh(z1)
    z2 = a1 @ layer[1]
    a2 = np.tanh(z2)
    z3 = a2 @ layer[2]
    a3 = softmax(z3)
    return a3

def tanhDiff(x):
    return 1 - np.tanh(x)**2

def softmaxDiff(x):
    s = x.reshape(-1, 1)  # 列ベクトルに変換
    return np.diagflat(s) - np.dot(s, s.T)  # ヤコビ行列

def update(y, t, eta=0.01):
    L = np.log(y)*t # Loss by entropy
    a3_b = np.sum(softmaxDiff(L))
    z3_b = layer[2] @ a3_b
    a2_b = tanhDiff(z3_b)
    z2_b = layer[1] @ a2_b
    a1_b = tanhDiff(z2_b)
    z1_b = layer[0] @ a1_b
    
    layer[0] += eta*z1_b
    layer[1] += eta*z2_b
    layer[2] += eta*z3_b

for i in range(2, X_train.shape[1]):
    NNoutput = forward(X_train[i])
    print(NNoutput)
    update(NNoutput, y_train[i])


maxindex = np.argmax(NNoutput, axis=1)
modelOutput = []
for i in maxindex:
    modelOutput.append(v[i])

print(modelOutput)

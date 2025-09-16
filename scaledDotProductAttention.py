import numpy as np

def softmax(x):
    x = x - np.max(x, axis=-1, keepdims=True)
    return np.exp(x) / np.sum(np.exp(x), axis=-1, keepdims=True) 

class ScaledDotProductAttention:
    def __init__(self, dmodel=8, dk=64, dv=64):
        self.dk = dk
        self.dv = dv
        rng = np.random.default_rng()
        self.Wq = rng.random((dmodel, dk))
        self.Wk = rng.random((dmodel, dk))
        self.Wv = rng.random((dmodel, dv))
    
    def selfQKV(self, X: np.array):
        self.q = X @ self.Wq
        self.k = X @ self.Wk
        self.v = X @ self.Wv
    
    def crossQKV(self, X, Y):
        self.q = X @ self.Wq
        self.k = Y @ self.Wk
        self.v = Y @ self.Wv
    
    def scaledDotProduct(self):
        matmul = self.q @ self.k.T
        scaled = matmul / np.sqrt(self.dk)
        softmaxed = softmax(scaled)
        output = softmaxed @ self.v
        return output

if __name__ == "__main__":
    dmodel = 10
    X = np.random.default_rng().random((20, dmodel))
    sdpa = ScaledDotProductAttention(dmodel, 15, 25)
    sdpa.selfQKV(X)
    print(sdpa.scaledDotProduct())
        
        
        
        
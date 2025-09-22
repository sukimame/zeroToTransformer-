import numpy as np

class FeedforwardNetwork:
    def __init__(self, dmodel=512, dk=64, dv=64, dff=2048):
        rng = np.random.default_rng()
        
        self.W1 = rng.random((dmodel, dff))
        self.W2 = rng.random((dff, dmodel))
        self.b1 = rng.random((dff))
        self.b2 = rng.random((dmodel))
    
    def forward(self, X):
        z = X @ self.W1 + self.b1
        a = np.maximum(0, z)
        output = a @ self.W2 + self.b2
        return output

if __name__ == "__main__":
    rng = np.random.default_rng()
    X = rng.random((64, 512))
    
    ffn = FeedforwardNetwork()
    print(ffn.forward(X).shape)
    
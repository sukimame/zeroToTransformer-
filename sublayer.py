from multiHeadAttention import MultiHeadAttention
from feedforwardNetwork import FeedforwardNetwork
import numpy as np

class sublayer:
    def __init__(self, dmodel=512):
        self.layerType = None
        self.proc = None
        
        self.gamma = np.ones((dmodel, ))
        self.beta = np.zeros((dmodel, ))
        
    def LayerNorm(self, X):
        mu = np.mean(X, axis=-1, keepdims=True)
        var = np.var(X, axis=-1, keepdims=True)
        sigma = np.sqrt(var + 1e-6)
        x = (X - mu)/sigma
        y = self.gamma * x + self.beta
        return y
        
    def fFNsublayer(self):
        self.layerType = FeedforwardNetwork()
        self.proc = self.layerType.forward
        
    def selfMHAsublayer(self):
        self.layerType = MultiHeadAttention()
        self.proc = self.layerType.selfAttention
    
    def crossMHAsublayer(self):
        self.layerType = MultiHeadAttention()
        self.proc = self.layerType.crossAttention
        
    def layerforward(self, X, Y=None):
        if Y is None:
            return self.LayerNorm(X + self.proc(X))
        else:
            return self.LayerNorm(X + self.proc(X, Y))
            
if __name__ == "__main__":
    rng = np.random.default_rng()
    X = rng.random((64, 512))
    Y = rng.random((64, 512))
    
    s1 = sublayer()
    s1.fFNsublayer()
    s2 = sublayer()
    s2.selfMHAsublayer()
    s3 = sublayer()
    s3.crossMHAsublayer()
    print(s1.layerforward(X).shape)
    print(s2.layerforward(X).shape)
    print(s3.layerforward(X, Y).shape)
    
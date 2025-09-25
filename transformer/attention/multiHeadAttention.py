from .scaledDotProductAttention import ScaledDotProductAttention
import numpy as np

class MultiHeadAttention:
    def __init__(self, h=8, dmodel=512, dk=64, dv=64):
        self.h = h # number of heads
        self.Ws = [] # liner transformation to add weight in keys, quelis and values for each head 
        self.multiHead = []
        
        rng = np.random.default_rng()
        for _ in range(self.h):
            s = ScaledDotProductAttention(dmodel, dk, dv)
            self.multiHead.append(s)
            self.Ws.append(rng.random((dmodel, dmodel)))

        self.Wo = rng.random((h*dv, dmodel))

    def selfAttention(self, X):
        headOutputs = []
        for i, head in enumerate(self.multiHead):
            head.selfQKV(X @ self.Ws[i])
            output = head.scaledDotProduct()
            headOutputs.append(output)
        concatMat = np.concatenate(headOutputs, axis=-1)
        
        return concatMat @ self.Wo
    
    def crossAttention(self, X, Y): # decoder-encoder Attention
        headOutputs = []
        for head in self.multiHead:
            head.crossQKV(X, Y)
                        
            output = head.scaledDotProduct()
            headOutputs.append(output)
        concatMat = np.concatenate(headOutputs, axis=-1)
        
        return concatMat @ self.Wo

if __name__ == "__main__":
    rng = np.random.default_rng()
    X = rng.random((64, 512))
    
    s = MultiHeadAttention(2)
    print(s.selfAttention(X).shape)


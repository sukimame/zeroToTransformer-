from scaledDotProductAttention import ScaledDotProductAttention
import numpy as np

class MultiHeadAttention:
    def __init__(self, h=8, dmodel=512, dk=64, dv=64):
        self.h = h
        self.multiHead = []
        for _ in range(self.h):
            s = ScaledDotProductAttention(dmodel, dk, dv)
            self.multiHead.append(s)
        
        rng = np.random.default_rng()
        self.Wo = rng.random((h*dv, dmodel))

    def multiHeadAttention(self, X):
        headOutputs = []
        for head in self.multiHead:
            head.computeQKV(X)
            output = head.scaledDotProduct()
            headOutputs.append(output)
        concatMat = np.concatenate(headOutputs, axis=-1)
        
        return concatMat @ self.Wo

if __name__ == "__main__":
    rng = np.random.default_rng()
    X = rng.random((64, 512))
    
    s = MultiHeadAttention(2)
    print(s.multiHeadAttention(X).shape)


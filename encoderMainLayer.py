from sublayer import SubLayer
import numpy as np

class EncoderMainLayer:
    def __init__(self, dmodel=512):
        self.l1 = SubLayer()
        self.l1.selfMHAsublayer()
        self.l2 = SubLayer()
        self.l2.fFNsublayer()
    
    def mainLayerforward(self, X):
        z1 = self.l1.layerforward(X)
        z2 = self.l2.layerforward(z1)
        return z2

if __name__ == "__main__":
    layer = EncoderMainLayer()
    rng = np.random.default_rng()
    X = rng.random((64, 512))
    
    z = layer.mainLayerforward(X)
    print(z.shape)
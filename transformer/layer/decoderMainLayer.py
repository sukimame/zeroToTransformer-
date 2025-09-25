from .sublayer import SubLayer
import numpy as np

class DecoderMainLayer:
    def __init__(self, dmodel=512):
        self.l1 = SubLayer()
        self.l1.selfMHAsublayer()
        self.l2 = SubLayer()
        self.l2.crossMHAsublayer()
        self.l3 = SubLayer()
        self.l3.fFNsublayer()
    
    def mainLayerforward(self, X, Y):
        z1 = self.l1.layerforward(X)
        z2 = self.l2.layerforward(z1, Y)
        z3 = self.l3.layerforward(z2)
        return z3

if __name__ == "__main__":
    layer = DecoderMainLayer()
    rng = np.random.default_rng()
    X = rng.random((64, 512))
    Y = rng.random((64, 512))
    
    z = layer.mainLayerforward(X, Y)
    print(z.shape)
    
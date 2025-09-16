from decoderMainLayer import DecoderMainLayer
from encoderMainLayer import EncoderMainLayer
import numpy as np

def softmax(x):
    x = x - np.max(x, axis=-1, keepdims=True)
    return np.exp(x) / np.sum(np.exp(x), axis=-1, keepdims=True)

class model:
    def __init__(self, v_len, dmodel=512):
        self.encoder = [EncoderMainLayer() for _ in range(6)]
        self.decoder = [DecoderMainLayer() for _ in range(6)]
        
        rng = np.random.default_rng()
        self.W = rng.random((512, v_len))
    
    def forward(self, inputs, outputs):
        z_en = inputs
        z_de = outputs
        for layer in self.encoder:
            z_en = layer.mainLayerforward(z_en)

        for layer in self.decoder:
            z_de = layer.mainLayerforward(z_de, z_en)
        
        logits = z_de @ self.W
        return logits
    
if __name__ == "__main__":
    inputs = np.ones((1, 512))
    outputs = np.ones((1, 512))
    m = model(1000)
    print(m.forward(inputs, outputs).shape)
        

    
    
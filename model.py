from decoderMainLayer import DecoderMainLayer
from encoderMainLayer import EncoderMainLayer
from inputProc import InputProc
import numpy as np

def softmax(x):
    x = x - np.max(x, axis=-1, keepdims=True)
    return np.exp(x) / np.sum(np.exp(x), axis=-1, keepdims=True)

class model:
    def __init__(self, v_len, dmodel=512):
        self.encoder = [EncoderMainLayer() for _ in range(6)]
        self.decoder = [DecoderMainLayer() for _ in range(6)]
        
        self.dmodel = dmodel
        
        rng = np.random.default_rng()
        self.W = rng.random((512, v_len))
    
    def forward(self, inputs, outputs):
        z_en = inputs + InputProc.positional_encoding(*inputs.shape)
        z_de = outputs + InputProc.positional_encoding(*outputs.shape)
        for layer in self.encoder:
            z_en = layer.mainLayerforward(z_en)

        for layer in self.decoder:
            z_de = layer.mainLayerforward(z_de, z_en)
        
        logits = z_de @ self.W
        return logits
    
if __name__ == "__main__":
    inputs = np.ones((64, 512))
    outputs = np.ones((64, 512))
    m = model(1000)
    print(m.forward(inputs, outputs).shape)
        

    
    
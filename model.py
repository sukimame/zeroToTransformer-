from decoderMainLayer import DecoderMainLayer
from encoderMainLayer import EncoderMainLayer
from inputProc import InputProc
import numpy as np

def softmax(x):
    x = x - np.max(x, axis=-1, keepdims=True)
    return np.exp(x) / np.sum(np.exp(x), axis=-1, keepdims=True)

class Model:
    def __init__(self, dmodel=512):
        self.encoder = [EncoderMainLayer() for _ in range(6)]
        self.decoder = [DecoderMainLayer() for _ in range(6)]

        self.dmodel = dmodel
        
        self.Em = InputProc()
        rng = np.random.default_rng()
        self.W = rng.random((512, self.Em.sp.GetPieceSize()))
    
    def forward(self, inputs, outputs):
        
        inputs = self.Em.BPE(inputs)
        outputs = self.Em.BPE(outputs)
        
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
    m = Model()
    print(m.forward(inputs, outputs).shape)
        

    
    
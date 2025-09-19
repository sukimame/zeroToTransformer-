import numpy as np
import sentencepiece as spm

class InputProc:
    def __init__(self):
        self.sp = spm.SentencePieceProcessor()
        self.sp.Load("sentencepiece.model")
        
        vocab_size = self.sp.GetPieceSize()
        d_model = 512
        rng = np.random.default_rng()
        self.embedding_matrix = rng.normal(0, 0.01, (vocab_size, d_model))
    
    def positional_encoding(seq_len, d_model):
        pos = np.arange(seq_len)[:, np.newaxis]              # (seq_len, 1)
        i = np.arange(d_model)[np.newaxis, :]                # (1, d_model)
        
        angle_rates = 1 / np.power(10000, (2 * (i//2)) / d_model)
        angles = pos * angle_rates
    
        PE = np.zeros((seq_len, d_model))
        PE[:, 0::2] = np.sin(angles[:, 0::2])
        PE[:, 1::2] = np.cos(angles[:, 1::2])
        return PE
    
    def BPE(self, X):
        ids = self.sp.EncodeAsIds(X)
        
        Y = []
        Y.append(self.embedding_matrix[ids])
        return np.array(*Y)

if __name__ == "__main__":
    i = InputProc()
    print(i.BPE("きみにちょっとしたものをもってきたよ。"))
    

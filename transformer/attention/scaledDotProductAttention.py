import numpy as np

def softmax(x):
    x = x - np.max(x, axis=-1, keepdims=True)
    return np.exp(x) / np.sum(np.exp(x), axis=-1, keepdims=True)

def diff_softmax(y):
    _, l = y.shape
    delta = np.eye(l)[None, :, :] * y[:, :, None]
    outer = y[:, :, None] * y[:, None, :]
    diff = delta - outer
    return diff

class ScaledDotProductAttention:
    def __init__(self, dmodel=512, dk=64, dv=64):
        self.dk = dk
        self.dv = dv
        rng = np.random.default_rng()
        self.Wq = rng.random((dmodel, dk))
        self.Wk = rng.random((dmodel, dk))
        self.Wv = rng.random((dmodel, dv))

    def selfQKV(self, X: np.array):
        self.q = X @ self.Wq
        self.k = X @ self.Wk
        self.v = X @ self.Wv

    def crossQKV(self, X, Y):
        self.q = X @ self.Wq
        self.k = Y @ self.Wk
        self.v = Y @ self.Wv

    def scaledDotProduct(self):
        matmul = self.q @ self.k.T
        scaled = matmul / np.sqrt(self.dk)
        self.softmaxed = softmax(scaled)
        output = self.softmaxed @ self.v
        return output

    def grad_sdp_attention(self, G):
        grad_V = self.softmaxed.T @ G
        grad_mat = G @ self.v.T
        grad_softmaxed = np.einsum('nij,ni->nj', diff_softmax(self.softmaxed), grad_mat)
        grad_scaled = grad_softmaxed * (1 / np.sqrt(self.dk))
        grad_Q = grad_scaled @ self.k
        grad_K = grad_scaled.T @ self.q
        return grad_V, grad_Q, grad_K

if __name__ == "__main__":
    dmodel = 10
    X = np.random.default_rng().random((20, dmodel))
    sdpa = ScaledDotProductAttention(dmodel, 15, 25)
    sdpa.selfQKV(X)
    G = sdpa.scaledDotProduct()
    g_V, g_Q, g_K = sdpa.grad_sdp_attention(G)
    print(g_V.shape, g_Q.shape, g_K.shape)

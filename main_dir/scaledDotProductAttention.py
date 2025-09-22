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
        grad_softmaxed = np.einsum('nij,nj->ni', diff_softmax(self.softmaxed), grad_mat)
        grad_scaled = grad_softmaxed * 1/np.sqrt(self.dk)
        grad_Q = grad_scaled @ self.k
        grad_K = grad_scaled.T @ self.q
        
        return grad_V, grad_Q, grad_K
import numpy as np

def compute_analytic_weight_grads(sdpa, X, target):
    # 前向き
    sdpa.selfQKV(X)
    out = sdpa.scaledDotProduct()   # shape (n, dv)

    # 損失 L = sum((out - target)^2)
    # dL/dout = 2*(out - target)
    G = 2.0 * (out - target)

    # dL/dV, dL/dQ, dL/dK を得る
    grad_V, grad_Q, grad_K = sdpa.grad_sdp_attention(G)  # shapes (n,dv),(n,dk),(n,dk)

    # 重み行列への勾配: X.T @ grad_*
    # X: (n, dmodel) ; grad_*: (n, dX) -> X.T @ grad_*(shape: (dmodel, dX))
    dWv = X.T @ grad_V   # shape (dmodel, dv)
    dWq = X.T @ grad_Q   # shape (dmodel, dk)
    dWk = X.T @ grad_K   # shape (dmodel, dk)

    return dWq, dWk, dWv

def numerical_grad_weights(sdpa, X, target, eps=1e-5):
    grads_num = {}
    for name in ['Wq', 'Wk', 'Wv']:
        W = getattr(sdpa, name)
        num_grad = np.zeros_like(W)
        it = np.nditer(W, flags=['multi_index'], op_flags=['readwrite'])
        while not it.finished:
            idx = it.multi_index
            orig = W[idx]

            # +eps
            W[idx] = orig + eps
            sdpa.selfQKV(X)
            out1 = sdpa.scaledDotProduct()
            loss1 = np.sum((out1 - target)**2)

            # -eps
            W[idx] = orig - eps
            sdpa.selfQKV(X)
            out2 = sdpa.scaledDotProduct()
            loss2 = np.sum((out2 - target)**2)

            num_grad[idx] = (loss1 - loss2) / (2*eps)

            # restore
            W[idx] = orig
            it.iternext()

        grads_num[name] = num_grad
    return grads_num

def compare_grads(sdpa, X, target, eps=1e-5):
    # analytic
    dWq_a, dWk_a, dWv_a = compute_analytic_weight_grads(sdpa, X, target)
    # numerical
    grads_num = numerical_grad_weights(sdpa, X, target, eps=eps)

    # 比較指標（絶対誤差ノルムと相対誤差）
    for name, dW_a in [('Wq', dWq_a), ('Wk', dWk_a), ('Wv', dWv_a)]:
        dW_n = grads_num[name]
        abs_err = np.linalg.norm(dW_n - dW_a)
        rel_err = abs_err / (np.linalg.norm(dW_n) + np.linalg.norm(dW_a) + 1e-12)
        max_abs = np.max(np.abs(dW_n - dW_a))
        print(f"{name}: abs_norm_diff={abs_err:.6e}, rel_err={rel_err:.6e}, max_abs={max_abs:.6e}")
    return

# ====== 実行例 ======
if __name__ == "__main__":
    np.random.seed(0)
    dmodel, dk, dv = 6, 4, 3
    n = 5
    X = np.random.randn(n, dmodel)
    target = np.random.randn(n, dv)

    sdpa = ScaledDotProductAttention(dmodel, dk, dv)
    # 乱数シードを合わせたいなら sdpa の乱数元も固定して初期化し直してください

    compare_grads(sdpa, X, target, eps=1e-5)


"""
if __name__ == "__main__":
    dmodel = 10
    X = np.random.default_rng().random((20, dmodel))
    sdpa = ScaledDotProductAttention(dmodel, 15, 25)
    sdpa.selfQKV(X)
    G = sdpa.scaledDotProduct()
    g_V, g_Q, g_K = sdpa.grad_sdp_attention(G)
    print(g_V.shape, g_Q.shape, g_K.shape)
"""
        
        
        
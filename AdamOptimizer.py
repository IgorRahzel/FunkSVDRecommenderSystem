import numpy as np

class AdamOptimizer:
    def __init__(self,lr=0.01, beta1=0.9, beta2=0.999, epsilon=1e-8):
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.t = 0

    def step(self,m,v,grads):
        self.t += 1
        m = self.beta1 * m + (1 - self.beta1) * grads
        v = self.beta2 * v + (1 - self.beta2) * (grads**2)
        m_hat = m/(1-(self.beta1**self.t))
        v_hat = v/(1-(self.beta2**self.t))
        update = -self.lr * (m_hat)/(np.sqrt(v_hat)+self.epsilon)
        return update, m, v
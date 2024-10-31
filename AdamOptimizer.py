import numpy as np

class AdamOptimizer:
    def __init__(self,lr=0.01, beta1=0.9, beta2=0.999, epsilon=1e-8):
        
        self.lr = lr                # Learning rate 
        self.beta1 = beta1          # Decay rate for momentum  
        self.beta2 = beta2          # Decaay rate for RMSProp term
        self.epsilon = epsilon      # Small constant to avoid divisions by zero
        self.t = 0                  # Time step counter

    def step(self,m,v,grads):

        self.t += 1 # Increment time step counter

        # Updating first and second moments
        m = self.beta1 * m + (1 - self.beta1) * grads
        v = self.beta2 * v + (1 - self.beta2) * (grads**2)
        
        # Compute bias correction for first and second moment
        m_hat = m/(1-(self.beta1**self.t))
        v_hat = v/(1-(self.beta2**self.t))

        # Compute the parameter update
        update = -self.lr * (m_hat)/(np.sqrt(v_hat)+self.epsilon)

        # Return the update for the weights, and the updated moments m and v
        return update, m, v
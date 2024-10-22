import numpy as np
import pandas as pd
from MatrixBuilder import MatrixBuilder

builder = MatrixBuilder()
df = builder()
df = df.fillna(0)
A = df.values
print(A.shape)
U,S,Vt = np.linalg.svd(A,full_matrices=False)
k= 415
print(f"sigma = {len(S)}")
U = U[:,:k]
S = np.diag(S[:k])
Vt= Vt[:k,:]
print(df)
prediction = U@S@Vt
print(prediction)
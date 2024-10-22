import pandas as pd
import numpy as np
from MatrixBuilder import MatrixBuilder



print(pd.__version__)
print(np.__version__)

#Building User-Item matrix for train and test sets
builder = MatrixBuilder()
user_item_matrix = builder()
user_item_matrix = user_item_matrix.fillna(0)
print(user_item_matrix)
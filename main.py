import pandas as pd
import numpy as np
from TrainTestSplit import TrainTestSplit
from FunkSVD import FunkSVD



print(pd.__version__)
print(np.__version__)

splitter = TrainTestSplit()
train_data, test_data = splitter()

model = FunkSVD(train_data)
model.train()

#TEST PREDICTIONS ON TRAINING SET
#TEST PREDICTION ON TEST SET

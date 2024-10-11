import pandas as pd
import numpy as np
from MatrixBuilder import MatrixBuilder
from Normalizer import Normalizer
from SimilarityCalculator import SimilarityCalculator

#print(pd.__version__)
#print(np.__version__)

#Building User-Item matrix for train and test sets
builder = MatrixBuilder()
train_set, test_set = builder()
print(train_set.head())

#Normalizing training matrix
normalizer = Normalizer()
user_item_matrix, user_mean = normalizer(train_set)
print(user_item_matrix)

#Building Similarity Matrix
similarity = SimilarityCalculator()
similarity_matrix = similarity(user_item_matrix)
print(similarity_matrix)




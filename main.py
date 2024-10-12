import pandas as pd
import numpy as np
from MatrixBuilder import MatrixBuilder
from Normalizer import Normalizer
from SimilarityCalculator import SimilarityCalculator
from RatePredictor import RatePredictor

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

#Predict Ratings
predictor = RatePredictor(user_item_matrix,similarity_matrix,user_mean)


#select a random user and item:
def select_random_rated_user_item(user_item_matrix, seed=None):
    # Set seed if provided for reproducibility (optional)
    if seed is not None:
        np.random.seed(seed)

    # Select a random user index from the user-item matrix
    random_user_index = np.random.choice(user_item_matrix.shape[0])
    random_user = user_item_matrix.index[random_user_index]
    
    # Get all items rated by the selected user (value > 0)
    rated_items = user_item_matrix.iloc[random_user_index]
    rated_items_indices = rated_items[rated_items > 0].index
    
    # If no items are rated by this user, try another user
    if len(rated_items_indices) == 0:
        return select_random_rated_user_item(user_item_matrix, seed)
    
    # Select a random item from the rated items
    random_item = np.random.choice(rated_items_indices)
    
    return random_user, random_item

# Example usage:
# Assuming user_item_matrix is a pandas DataFrame with users as rows and items as columns
# user_item_matrix = pd.DataFrame(...)  # Define your matrix here

for i in range(10):
    random_user, random_item = select_random_rated_user_item(train_set)
    rating = train_set.loc[random_user,random_item]
    print(f"Random user: {random_user}, Random item: {random_item}, Rating: {rating}")

    yhat = predictor.topK_neighbors(random_item,random_user)
    print(yhat)
    print("------------------------------------------\n\n")
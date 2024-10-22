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
unomarlized_user_item = builder()

#Normalizing training matrix
normalizer = Normalizer()
user_item_matrix, user_mean = normalizer(unomarlized_user_item)

'''
ratings_builder = MatrixBuilder()
df_ratings = ratings_builder._pre_processing()
user_ratings = df_ratings[df_ratings["UserId"]=='afced32639']
print(user_ratings["Rating"].mean())

print(user_mean['afced32639'])
'''

#Building Similarity Matrix
similarity = SimilarityCalculator()
similarity_matrix = similarity(user_item_matrix)

#Raw data
ratings_builder = MatrixBuilder()
df_ratings = ratings_builder._pre_processing()

#Predict Ratings
predictor = RatePredictor(user_item_matrix,similarity_matrix,user_mean,df_ratings)

target_file = 'targets.csv'
targets_builder = MatrixBuilder(data=target_file)
df_target = targets_builder._pre_processing()

# Abre o arquivo output.csv para escrita
with open('output.csv', 'w') as file:
    # Escreve o cabeçalho
    file.write('UserId:ItemId,Rating\n')

    # Itera pelas linhas do df_target e escreve cada linha no CSV
    for _, row in df_target.iterrows():
        user_id = row['UserId']
        item_id = row['ItemId']
        rating = predictor.topK_neighbors(user_id,item_id)
        # Escreve a linha no arquivo
        file.write(f'{user_id}:{item_id},{rating}\n')

'''
#rating = predictor.topK_threshold

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


print("TEST MATRIX")
for i in range(20):
    j = np.random.randint(0,len(df_ratings))
    random_user = df_ratings['UserId'].iloc[j]
    random_item = df_ratings['ItemId'].iloc[j]
    rating = df_ratings['Rating'].iloc[j]
    print(f"Random user: {random_user}, Random item: {random_item}, Rating: {rating}")

    yhat = predictor.topK_threshold(random_user,random_item)
    print(yhat)
    print("------------------------------------------\n\n")
'''
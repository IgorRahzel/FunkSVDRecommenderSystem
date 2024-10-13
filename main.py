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
#print(unomarlized_user_item.head())

#Normalizing training matrix
normalizer = Normalizer()
user_item_matrix, user_mean = normalizer(unomarlized_user_item)
#print(user_item_matrix)
#print(user_item_matrix.index)

#Building Similarity Matrix
similarity = SimilarityCalculator()
similarity_matrix = similarity(user_item_matrix)
#print(similarity_matrix)


ratings_builder = MatrixBuilder()
df_ratings = ratings_builder._pre_processing()

#Predict Ratings
predictor = RatePredictor(user_item_matrix,similarity_matrix,user_mean,df_ratings)


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

'''
for i in range(10):
    random_user, random_item = select_random_rated_user_item(train_set)
    rating = train_set.loc[random_user,random_item]
    print(f"Random user: {random_user}, Random item: {random_item}, Rating: {rating}")

    yhat = predictor.topK_neighbors(random_item,random_user)
    print(yhat)
    print("------------------------------------------\n\n")
'''



target_file = 'targets.csv'
#df_target = pd.read_csv(target_file)
#df_target[['UserId', 'ItemId']] = df_target['UserId:ItemId'].str.split(':', expand=True)
#df_target = df_target.drop(columns=['UserId:ItemId'])
 


targets_builder = MatrixBuilder(data=target_file)
df_target = targets_builder._pre_processing()

#print(f"rating:\n {df_ratings.head()}")
#print(f"target:\n {df_target.head()}")

def count_user_ratings(df, user_id):
    # Filter the rows where the UserId matches the given user_id
    user_ratings = df[df['UserId'] == user_id]
    
    # Count the number of ratings (non-zero values) for that user
    rating_count = len(user_ratings)
    
    return rating_count

rating_count = count_user_ratings(df_ratings,'bc3b9136bc')

# Get unique users from both DataFrames
itens_in_ratings = set(df_ratings['ItemId'].unique())
itens_in_target = set(df_target['ItemId'].unique())
itens_in_user_item = set(user_item_matrix.columns)
itens_in_similarity = set(similarity_matrix.index)
# Check if all users in target.csv are present in ratings.csv
missing_itens = itens_in_target - itens_in_ratings
missing_itens2 = itens_in_ratings - itens_in_user_item
missing_itens3  = itens_in_ratings = itens_in_similarity
#print(f"missing items: {missing_itens}")
#print(f"missing items2: {missing_itens2}")
#print(f"missing items3: {missing_itens3}")

#if missing_itens2:
    #print("There are itens in target.csv that are not in ratings.csv:")
    #print(missing_itens)
#else:
    #print("All itens in target.csv are present in ratings.csv.")

print('UserId:ItemId,Rating')
for _, row in df_target.iterrows():
    user_id = row['UserId']
    item_id = row['ItemId']
    rating = predictor.topK_neighbors(item_id,user_id)
    print(f'{user_id}:{item_id},{rating}')

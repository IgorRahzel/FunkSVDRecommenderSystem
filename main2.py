import pandas as pd
import numpy as np

#print(pd.__version__)
#print(np.__version__)

#Reading the csv file
df = pd.read_csv('ratings.csv')

#Splitting UserID and ItemID into different columns
df[['UserId', 'ItemId']] = df['UserId:ItemId'].str.split(':', expand=True)

#Removing the UserId:ItemId column
df = df.drop(columns=['UserId:ItemId'])

centered_user_item_matrix, users_mean = matrix_builder()
similarity_matrix = cosine_similarity(centered_user_item_matrix)
predictions = predict_ratings()
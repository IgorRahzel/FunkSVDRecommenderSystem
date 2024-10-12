import pandas as pd
import numpy as np

class RatePredictor:
    def __init__(self,user_item_matrix,similarity_matrix,user_mean_rating):
        self.user_item_matrix = user_item_matrix
        self.similarity_matrix = similarity_matrix
        self.user_mean_rating = user_mean_rating
    

    def topK_neighbors(self,itemId, userId, k=5):
        # Get the similarities of other items with 'itemId'
        itemId_Similarities = self.similarity_matrix.loc[itemId].copy()
        
        # Remove the item itself from the similarities
        itemId_Similarities.drop(itemId, inplace=True)
        
        # Sort the similarities in descending order to get the most similar items first
        itemId_Similarities = itemId_Similarities.sort_values(ascending=False)
        
        # Filter the top K items with the highest similarity
        topK_similarities = itemId_Similarities.head(k)
        
        # Get the user’s ratings for the top K similar items
        user_ratings = self.user_item_matrix.loc[userId, topK_similarities.index]
        
        # Filter out items that the user has not rated
        rated_items = user_ratings[user_ratings > 0]

        # If the user has not rated any of the top K similar items, return a default value
        if rated_items.empty:
            # Optionally, return the average rating for the item or a default value
            return self.user_mean_rating[userId]
        
        # Get the similarities and the corresponding user ratings
        similarities = topK_similarities.loc[rated_items.index]
        ratings = rated_items
        
        # Calculate the weighted average rating
        predicted_rating = np.dot(similarities, ratings) / np.sum(np.abs(similarities))
        predicted_rating + self.user_mean_rating[userId]

        return predicted_rating

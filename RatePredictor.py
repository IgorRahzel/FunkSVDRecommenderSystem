import pandas as pd
import numpy as np

class RatePredictor:
    def __init__(self,user_item_matrix,similarity_matrix,user_mean_rating,raw_data):
        self.user_item_matrix = user_item_matrix
        self.similarity_matrix = similarity_matrix
        self.user_mean_rating = user_mean_rating
        self.raw_data = raw_data
    

    def topK_neighbors(self,userId,itemId, k=3):
        # Handling cases where user is not in the user-item matrix
        if userId not in self.user_item_matrix.index:
            item_raw_data = self.raw_data[self.raw_data['ItemId'] == itemId]
    
            # Check if there are any ratings for the item
            if not item_raw_data.empty:
                item_avg_rating = item_raw_data['Rating'].mean()
                return item_avg_rating
            else:
                # If there are no ratings for the item, return a global mean or some default value
                global_mean_rating = self.raw_data['Rating'].mean()
                return global_mean_rating
        

        # Handling cases where item is not in the user-item matrix
        if itemId not in self.user_item_matrix.columns:
            # Fetch all ratings for the item
            item_raw_data = self.raw_data[self.raw_data['ItemId'] == itemId]

            # Check if there are any ratings for the item
            if not item_raw_data.empty:
                item_avg_rating = item_raw_data['Rating'].mean()
                return item_avg_rating
            else:
                # If there are no ratings for the item, return a global mean or some default value
                global_mean_rating = self.raw_data['Rating'].mean()
                return global_mean_rating

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


    def topK_threshold(self,userId,itemId,k = 5, threshold = 0):
        #se o item ou usuario não estiverem na user_item matrix retorna media do item
        #caso contrario:
        #   -seleciona a linha contendo as similaridades do item 'itemId'
        #   -filtra linha selecionada para conter apenas similaridades > threshold
        #   -ordena a linha em ordem decrescente
        #   -seleciona apenas as top-K similaridades
        #   -obtem o idx('ItemId') correspondente a cada uma das topK similaridades
        #   -seleciona coluna da user-item matrix referente ao usuario 'userId'
        #   -filtra linha para conter apenas os idx selecionados anteriormente
        #   -ordena os ratings do usuario de acordo com o array idx
        #   -basta então obter a previsão realizando o dot product ente a linha de usuarios e items e então dividir o resultado pela soma absoluta das similatidades
        #handling cases of missing itemId or missing userId
        if itemId not in self.user_item_matrix.columns or userId not in self.user_item_matrix.index:
            item_raw_data = self.raw_data[self.raw_data['ItemId']==itemId]
            return item_raw_data['Rating'].mean()
        #Making predictions
        else:
            # Filter similarity values for the given itemId
            filtered_item_similarities = self.similarity_matrix.loc[itemId].copy()
            # Remove the item itself from the similarities
            filtered_item_similarities.drop(itemId, inplace=True)
            # Filtering similarities with respect to the threshold
            filtered_item_similarities = filtered_item_similarities[filtered_item_similarities > threshold]
            
            # Sort and select top-k similar items
            filtered_item_similarities = filtered_item_similarities.sort_values(ascending=False).head(k)
            
            # Extract item indexes
            similar_items_idx = filtered_item_similarities.index
            
            # Get the ratings given by the user for these similar items
            user_ratings = self.user_item_matrix.loc[userId, similar_items_idx]
            
            # If there are no ratings available, return the user's mean rating
            if len(user_ratings) == 0:
                return self.user_mean_rating[userId]
            
            # Compute the weighted average prediction
            prediction = np.dot(filtered_item_similarities.values, user_ratings) / np.sum(np.abs(filtered_item_similarities))
            return prediction + self.user_mean_rating[userId]
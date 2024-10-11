import pandas as pd
import numpy as np

class SimilarityCalculator:
    def __init__(self, metric = 'cosine'):
        self.metric = metric
    
    def __call__(self,user_item_matrix):
        #Normalize user-item-matrix
        mean,user_item_matrix = self._normalize_matrix(user_item_matrix)
        #calculate similarity based on the chosen metric
        if self.metric == 'cosine':
            return self._compute_cosine_similarity(user_item_matrix)
        
        elif self.metric == 'pearson':
            self._compute_pearson_similarity(user_item_matrix)
        
        elif self.metric == 'adjusted_cosine':
            self._compute_adjusted_cosine_similarity(user_item_matrix)
        
        else:
            raise ValueError("Unsupported metric")
    
    def _normalize_matrix(self,user_item_matrix):
        #storing users mean rating
        users_mean_rating = user_item_matrix.mean(axis=1,skipna=True)
        # Subtract the mean rating per user, ignoring NaNs to only center on rated items
        user_item_matrix = user_item_matrix.fillna(0)
        user_item_matrix = user_item_matrix.subtract(users_mean_rating, axis=0)
        return users_mean_rating, user_item_matrix
    
    def _compute_cosine_similarity(self, user_item_matrix):
        # Implementation for cosine similarity
        # Normalize the user-item matrix (L2 normalization for each column)
        norms = np.sqrt((user_item_matrix ** 2).sum(axis=0))
        normalized_matrix = user_item_matrix / norms

        # Fill NaNs that may result from zero divisions (if any norms are zero, though this should be handled above)
        normalized_matrix = normalized_matrix.fillna(0)

        # Compute the cosine similarity as the dot product of the normalized matrix
        cosine_similarity_matrix = normalized_matrix.T.dot(normalized_matrix)

        # Convert the result to a DataFrame for easier manipulation and to retain labels
        cosine_similarity_df = pd.DataFrame(cosine_similarity_matrix, index=user_item_matrix.columns, columns=user_item_matrix.columns)
        return cosine_similarity_df
    
    def _compute_pearson_similarity(self, user_item_matrix):
        # Implementation for Pearson correlation
        pass

    def _compute_adjusted_cosine_similarity(self,user_item_matrix):
        # Implementation for Pearson correlation
        pass
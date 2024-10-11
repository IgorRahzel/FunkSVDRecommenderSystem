import numpy as np
import pandas as pd

class Normalizer:
    def __init__(self,norm_type = 'mean'):
        self.norm_type = norm_type
    
    def __call__(self,user_item_matrix):
        if self.norm_type == 'mean':
            return self.mean_norm(user_item_matrix)
            
        else:
            raise ValueError("Unsupported normalization")

    def mean_norm(self,user_item_matrix):
        #storing users mean rating
        users_mean_rating = user_item_matrix.mean(axis=1,skipna=True)

        # Subtract the mean rating per user, ignoring NaNs to only center on rated items
        user_item_matrix = user_item_matrix.fillna(0)
        user_item_matrix = user_item_matrix.subtract(users_mean_rating, axis=0)

        return user_item_matrix, users_mean_rating

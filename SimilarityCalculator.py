import pandas as pd
import numpy as np

class SimilarityCalculator:
    def __init__(self, metric = 'cosine'):
        self.metric = metric
    
    def __call__(self,user_item_matrix):
        #calculate similarity based on the chosen metric
        if self.metric == 'cosine':
            self._compute_cosine_similarity(user_item_matrix)
        
        elif self.metric == 'pearson':
            self._compute_pearson_similarity(user_item_matrix)
        
        elif self.metric == 'adjusted_cosine':
            self._compute_adjusted_cosine_similarity(user_item_matrix)
        
        else:
            raise ValueError("Unsupported metric")
        
    
    def _compute_cosine_similarity(self, user_item_matrix):
        # Implementation for cosine similarity
        pass
    
    def _compute_pearson_similarity(self, user_item_matrix):
        # Implementation for Pearson correlation
        pass

    def _compute_adjusted_cosine_similarity(self,user_item_matrix):
        # Implementation for Pearson correlation
        pass
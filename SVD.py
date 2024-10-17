import numpy as np
import pandas as pd

class SVD:
    def __init__(self, user_item_matrix, k):
        """
        Initialize the SVD model.
        :param user_item_matrix: The user-item matrix as a Pandas DataFrame
        :param k: The number of singular values to keep (for truncation)
        """
        self.user_item_matrix = user_item_matrix
        self.user_ids = user_item_matrix.index
        self.item_ids = user_item_matrix.columns

        self.k = k

        self.U = None
        self.Sigma = None
        self.Vt = None

        self.decompose()
        self.predction_matrix = pd.DataFrame((self.U * self.Sigma) @ self.Vt , index=self.user_ids, columns=self.item_ids)
      
    
    def __call__(self,user_id,item_id): 
        self.predict(user_id,item_id)
        

    def decompose(self):
        """
        Decompose the user-item matrix using SVD and truncate to k singular values.
        """
        # Convert DataFrame to NumPy array
        matrix = self.user_item_matrix.fillna(0).values

        # Compute SVD
        U,S,Vt = np.linalg.svd(matrix, full_matrices=False)

        # Truncate U, S, and Vt to keep only the first k singular values
        self.U = U[:, :self.k]
        self.S = np.diag(S[:self.k])
        self.Vt = Vt[:self.k, :]


    def predict(self, user_id, item_id):
        """
        Predict the rating for a specific user-item pair using the user and item IDs.
        :param user_id: The ID of the user
        :param item_id: The ID of the item
        :return: The predicted rating
        """
        
        # Check if the user and item exist in the matrix
        if user_id in self.predicted_ratings.index and item_id in self.predicted_ratings.columns:
            return self.predicted_ratings.loc[user_id, item_id]
        else:
            return 2.5  # Return NaN if the user or item is not in the matrix

import numpy as np
import pandas as pd

class FunkSVD:
    def __init__(self, dataframe):
        self.dataframe = dataframe.copy()
        self.m = dataframe['UserId'].nunique() 
        self.n = dataframe['ItemId'].nunique()

        # Creating bias factors
        self.global_mean = dataframe['Rating'].mean()
        self.bu_vector = np.random.rand(self.m)
        self.bi_vector = np.random.rand(self.n)

        # Create a mapping for userId and itemId
        self.user_to_index = {user: idx for idx, user in enumerate(dataframe['UserId'].unique())}
        self.item_to_index = {item: idx for idx, item in enumerate(dataframe['ItemId'].unique())}

        # Add new columns for pre-mapped user and item indices
        self.dataframe.loc[:,'user_idx'] = self.dataframe['UserId'].map(self.user_to_index)
        self.dataframe.loc[:,'item_idx'] = self.dataframe['ItemId'].map(self.item_to_index)
    
    def _initializePQ(self, k):
        self.P = (np.sqrt(5/k)) * np.random.rand(self.m, k)
        self.Q = (np.sqrt(5/k)) * np.random.rand(k, self.n)
    
    def _getMiniBatch(self, batch_size):
        """Generate a mini-batch from the DataFrame."""
        shuffled_df = self.dataframe.sample(frac=1).reset_index(drop=True)  # Shuffle data
        for start in range(0, len(shuffled_df), batch_size):
            yield shuffled_df.iloc[start:start + batch_size]
        
    def _MiniBatchGradientDescent(self, k=100, batch_size=10, lr=0.01, lamda=0.02, epochs=20):
        # Initialize the P and Q matrices
        self._initializePQ(k)

        # Iterate over epochs
        for epoch in range(epochs):
            total_loss = 0

            # Iterate over mini-batches
            for batch in self._getMiniBatch(batch_size):
                users_idx = batch['user_idx'].to_numpy()
                items_idx = batch['item_idx'].to_numpy()

                # Make predictions for batch
                predictions = np.sum(self.P[users_idx, :] * self.Q[:, items_idx].T, axis=1) + self.global_mean + self.bi_vector[items_idx] + self.bu_vector[users_idx]
               
                error = batch['Rating'].to_numpy() - predictions

                # Reshape error for broadcasting
                error = error[:, np.newaxis]  # Shape (batch_size, 1)

                # Update P and Q matrices and b_u and b_i vectors
                P_new = self.P[users_idx, :] + lr * ((error * self.Q[:, items_idx].T) - (lamda * self.P[users_idx, :]))
                Q_new = self.Q[:, items_idx].T + lr * (error * self.P[users_idx, :] - lamda * self.Q[:, items_idx].T)
                bu_new = self.bu_vector[users_idx] + lr * (error.squeeze() - lamda * self.bu_vector[users_idx])
                bi_new = self.bi_vector[items_idx] + lr * (error.squeeze() - lamda * self.bi_vector[items_idx])


                self.P[users_idx, :] = P_new
                self.Q[:, items_idx] = Q_new.T
                self.bu_vector[users_idx] = bu_new
                self.bi_vector[items_idx] = bi_new

                 # Accumulate the squared error for loss calculation
                total_loss += np.sum(error**2)

            # Calculate total RMSE after each epoch
            avg_loss = np.sqrt(total_loss / len(self.dataframe))
            print(f"Epoch {epoch+1}/{epochs}, Loss (RMSE): {avg_loss}")

    def train(self, k=100, batch_size=10, lr=0.01, lamda=0.02, epochs=30):
        self._MiniBatchGradientDescent(k, batch_size, lr, lamda, epochs)
    
    def prediction(self, userId, itemId, item_mean):
        if userId not in self.user_to_index or itemId not in self.item_to_index:
            return item_mean[itemId]
        else:
            user_idx = self.user_to_index[userId]
            item_idx = self.item_to_index[itemId]
            return self.global_mean + self.bu_vector[user_idx] + self.bi_vector[item_idx] + self.P[user_idx, :] @ self.Q[:, item_idx]

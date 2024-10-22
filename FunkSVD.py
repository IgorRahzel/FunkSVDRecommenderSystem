import numpy as np
import pandas as pd

class FunkSVD:
    def __init__(self,dataframe):
        self.dataframe = dataframe
        self.m = dataframe['userId'].nunique() 
        self.n = dataframe['itemId'].nunique()

        # Create a mapping for userId and itemId
        self.user_to_index = {user: idx for idx, user in enumerate(dataframe['userId'].unique())}
        self.item_to_index = {item: idx for idx, item in enumerate(dataframe['itemId'].unique())}
    
    def _initializePQ(self,k):
        self.P = 5/np.sqrt(5) * np.random.randn(self.m,k)
        self.Q = 5/np.sqrt(5) * np.random.randn(k,self.n)
    
    def _getMiniBatch(self,batch_size):
        """Generate a mini-batch from the DataFrame."""
        shuffled_df = self.dataframe.sample(frac=1).reset_index(drop=True)  # Shuffle data
        for start in range(0, len(shuffled_df), batch_size):
            yield shuffled_df.iloc[start:start + batch_size]
        
    def _MiniBatchGradientDescent(self,k=100,batch_size = 10,lr = 0.01,lamda = 0.02, epochs = 30):
        #Initialize the P and Q matrices
        self._initializePQ(k)

        #Iterating over the epochs
        for epoch in range(epochs):
            total_loss = 0
            #Iterating over the batches:
            for batch in self._getMiniBatch(batch_size):
                users_idx = batch['userId'].map(self.user_to_index)
                items_idx = batch['itemId'].map(self.item_to_index)

                PredictedRatings = self.P[users_idx,:] @ self.Q[:,items_idx]
                
                error = batch['Rating'] - PredictedRatings

                P_new = self.P[users_idx,:] + lr*(error * self.Q[:,items_idx] - lamda * self.P[users_idx,:])
                Q_new = self.Q[:,items_idx] + lr*(error * self.P[users_idx,:] - lamda * self.Q[:,items_idx])

                self.P[users_idx,:] = P_new
                self.Q[:,items_idx] = Q_new

                # Calculate total loss (MSE)
                total_loss += np.sum(error**2)

            # Log the loss at the end of each epoch
            avg_loss = total_loss / len(self.dataframe)
            print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss}")




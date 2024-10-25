import numpy as np
import pandas as pd

class FunkSVD:
    def __init__(self,dataframe):
        self.dataframe = dataframe
        self.m = dataframe['UserId'].nunique() 
        self.n = dataframe['ItemId'].nunique()

        # Create a mapping for userId and itemId
        self.user_to_index = {user: idx for idx, user in enumerate(dataframe['UserId'].unique())}
        self.item_to_index = {item: idx for idx, item in enumerate(dataframe['ItemId'].unique())}
    
    def _initializePQ(self,k):
        self.P = (np.sqrt(5/k)) * np.random.rand(self.m,k)
        self.Q = (np.sqrt(5/k)) * np.random.rand(k,self.n)
    
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
            i = 0
            #Iterating over the batches:
            for batch in self._getMiniBatch(batch_size):
                users_idx = batch['UserId'].map(self.user_to_index).to_numpy()
                items_idx = batch['ItemId'].map(self.item_to_index).to_numpy()

                PredictedRatings = self.P[users_idx,:] @ self.Q[:,items_idx]
                
                #New mapping to acces predictions in the PredictedRatings matrix
                batch_user_to_index = {user: idx for idx, user in enumerate(batch['UserId'].unique())}
                batch_item_to_index = {item: idx for idx, item in enumerate(batch['ItemId'].unique())}
              
                predictions = []
                for _,row in batch.iterrows():
                    user = row['UserId']
                    item = row['ItemId']

                    batch_user_idx = batch_user_to_index[user]
                    batch_item_idx = batch_item_to_index[item]

                    prediction = PredictedRatings[batch_user_idx,batch_item_idx]
                    predictions.append(prediction)

                predictions = np.array(predictions)
                
                error = batch['Rating'].to_numpy() - predictions

                # Reshape error for broadcasting
                error = error[:, np.newaxis]  # Shape (batch_size, 1)

                P_new = self.P[users_idx,:] + lr*((error * self.Q[:,items_idx].T) - (lamda * self.P[users_idx,:]))
                Q_new = self.Q[:,items_idx].T + lr*(error * self.P[users_idx,:] - lamda * self.Q[:,items_idx].T)

                self.P[users_idx,:] = P_new
                self.Q[:,items_idx] = Q_new.T

                n = len(self.dataframe)
                iterations = n/batch_size

                print(f"iteration {i+1}  of {iterations}")
                i = i+1
                # Calculate total loss (MSE)
                total_loss += np.sum(error**2)

            # Log the loss at the end of each epoch
            avg_loss = total_loss / len(self.dataframe)
            print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss}")

    def train(self,k=100,batch_size = 10,lr = 0.01,lamda = 0.02, epochs = 30):
        self._MiniBatchGradientDescent(k,batch_size,lr,lamda , epochs)
    
    def prediction(self,userId,itemId,item_mean):
        if userId not in self.dataframe['UserId'] or itemId not in self.dataframe['ItemId']:
          return item_mean[itemId]
        else:
            user_idx = self.user_to_index[userId]
            item_idx = self.item_to_index[itemId]
            return self.P[user_idx,:] @ self.Q[:,item_idx]
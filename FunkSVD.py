import numpy as np
from AdamOptimizer import AdamOptimizer
class FunkSVD:
    def __init__(self, dataframe):

        self.dataframe = dataframe.copy()
        
        # Storing the number of unique users and items
        self.m = dataframe['UserId'].nunique() 
        self.n = dataframe['ItemId'].nunique()

        # Creating bias factors
        self.global_mean = dataframe['Rating'].mean()
        self.bu_vector = np.zeros(self.m)
        self.bi_vector = np.zeros(self.n)

        # Create a mapping for userId and itemId
        self.user_to_index = {user: idx for idx, user in enumerate(dataframe['UserId'].unique())}
        self.item_to_index = {item: idx for idx, item in enumerate(dataframe['ItemId'].unique())}

        # Add new columns for pre-mapped user and item indices
        self.dataframe.loc[:,'user_idx'] = self.dataframe['UserId'].map(self.user_to_index)
        self.dataframe.loc[:,'item_idx'] = self.dataframe['ItemId'].map(self.item_to_index)
    
    #Initialuzing P and Q
    def _initializePQ(self, k):
        self.P = (np.sqrt(5/k)) * np.random.rand(self.m, k)
        self.Q = (np.sqrt(5/k)) * np.random.rand(k, self.n)
    
    #Randomly creating the batches for each epoch
    def _getMiniBatch(self, batch_size):
        shuffled_df = self.dataframe.sample(frac=1).reset_index(drop=True)  # Shuffle data
        for start in range(0, len(shuffled_df), batch_size):
            yield shuffled_df.iloc[start:start + batch_size]
    
    #Training the Model using Mini-Batch Gradient Descent
    def _MiniBatchGradientDescent(self, k=100, batch_size=10, lr=0.01, lamda=0.02, epochs=20, momentum=0.9):
        # Initialize the P and Q matrices
        self._initializePQ(k)
        
        #Initializing the Adam Optimizer
        adam = AdamOptimizer()

        # Initialize velocities (momentum terms) for P, Q, bu, and bi
        vP = np.zeros_like(self.P)
        mP = np.zeros_like(self.P)

        vQ = np.zeros_like(self.Q)
        mQ = np.zeros_like(self.Q)

        vbu = np.zeros_like(self.bu_vector)
        mbu = np.zeros_like(self.bu_vector)

        vbi = np.zeros_like(self.bi_vector)
        mbi = np.zeros_like(self.bi_vector)

        # Iterate over epochs
        for epoch in range(epochs):
            total_loss = 0

            # Iterate over mini-batches
            for batch in self._getMiniBatch(batch_size):
                users_idx = batch['user_idx'].to_numpy()
                items_idx = batch['item_idx'].to_numpy()

                # Make predictions for the batch
                predictions = np.sum(self.P[users_idx, :] * self.Q[:, items_idx].T, axis=1) + self.global_mean + self.bi_vector[items_idx] + self.bu_vector[users_idx]
                error = batch['Rating'].to_numpy() - predictions

                # Reshape error for broadcasting
                error = error[:, np.newaxis]  # Shape (batch_size, 1)

                # Getting gradients of P, Q, bu and bi 
                P_grad = -((error * self.Q[:, items_idx].T) - (lamda * self.P[users_idx, :]))
                Q_grad = -(error * self.P[users_idx, :] - lamda * self.Q[:, items_idx].T)
                bu_grad = -(error.squeeze() - lamda * self.bu_vector[users_idx])
                bi_grad = -(error.squeeze() - lamda * self.bi_vector[items_idx])

                #Getting Update for each matrix and updating the momentums of each matrix
                updateP,mP[users_idx,:],vP[users_idx,:] = adam.step(mP[users_idx,:],vP[users_idx,:],P_grad)
                updateQ,mQ[:,items_idx],vQ[:,items_idx] = adam.step(mQ[:,items_idx],vQ[:,items_idx],Q_grad.T)
                updatebu,mbu[users_idx],vbu[users_idx] = adam.step(mbu[users_idx],vbu[users_idx],bu_grad)
                updatebi,mbi[items_idx],vbi[items_idx] = adam.step(mbi[items_idx],vbi[items_idx],bi_grad)

                #Updating weights
                P_new =  self.P[users_idx, :] + updateP
                Q_new = self.Q[:,items_idx] + updateQ
                self.bu_vector[users_idx] = self.bu_vector[users_idx] + updatebu
                self.bi_vector[items_idx] = self.bi_vector[items_idx] + updatebi

                self.P[users_idx, :] = P_new
                self.Q[:, items_idx] = Q_new

                # Accumulate the squared error for loss calculation
                total_loss += np.sum(error**2)

            # Calculate total RMSE after each epoch
            #avg_loss = np.sqrt(total_loss / len(self.dataframe))
            #print(f"Epoch {epoch+1}/{epochs}, Loss (RMSE): {avg_loss}")

    #Trainig the model
    def train(self, k=100, batch_size=10, lr=0.01, lamda=0.02, epochs=30):
        self._MiniBatchGradientDescent(k, batch_size, lr, lamda, epochs)
    
    #Making predictions
    def prediction(self, userId, itemId, item_mean):
        if userId not in self.user_to_index or itemId not in self.item_to_index:
            return item_mean[itemId]
        else:
            user_idx = self.user_to_index[userId]
            item_idx = self.item_to_index[itemId]
            prediction = self.global_mean + self.bu_vector[user_idx] + self.bi_vector[item_idx] + self.P[user_idx, :] @ self.Q[:, item_idx]
            return np.clip(prediction,1,5)
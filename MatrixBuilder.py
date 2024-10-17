import pandas as pd
import numpy as np

class MatrixBuilder:
    def __init__(self, user_col='UserId', item_col='ItemId', rating_col='Rating', data='ratings.csv'):
        self.user_col = user_col
        self.item_col = item_col
        self.rating_col = rating_col
        self.data = data

    def __call__(self):
        df = self._pre_processing()
        #train_data, test_data = self.split_data(df,random_state=13)
        return self.build_filtered_matrix(df)
    
    def _pre_processing(self):
        df = pd.read_csv(self.data)
        df[['UserId', 'ItemId']] = df['UserId:ItemId'].str.split(':', expand=True)
        df = df.drop(columns=['UserId:ItemId'])
        return df

    def split_data(self, data, test_size=0.2, random_state=None):
        if random_state:
            np.random.seed(random_state)
        # Generate a random array of the same length as the DataFrame
        random_values = np.random.uniform(0, 1, size=len(data))
        threshold = 1 - test_size
        # Split the DataFrame into training and test sets based on the threshold
        train_data = data[random_values < threshold]
        test_data = data[random_values >= threshold]
        return train_data, test_data

    def filter_users(self, data, min_interactions=5):
        user_counts = data[self.user_col].value_counts()
        filtered_users = user_counts[user_counts >= min_interactions].index
        filtered_data = data[data[self.user_col].isin(filtered_users)]
        return filtered_data

    def build_matrix(self, data):
        user_item_matrix = data.pivot_table(
            index=self.user_col,
            columns=self.item_col,
            values=self.rating_col,
        )
        return user_item_matrix

    def build_filtered_matrix(self, data, min_interactions=5):
        """
        Builds user-item matrices for training and testing sets after filtering users.

        Parameters:
        train_data (pd.DataFrame): Training data with user, item, and rating columns.
        test_data (pd.DataFrame): Testing data with user, item, and rating columns.
        min_interactions (int): Minimum number of interactions a user must have to be kept.

        Returns:
        tuple: Training and testing user-item matrices (pd.DataFrame, pd.DataFrame)
        """
        # Filter users in data 
        filtered_data = self.filter_users(data, min_interactions)
        # Build matrix
        matrix = self.build_matrix(filtered_data)
       
        
        return matrix


#builder = MatrixBuilder()
#train_matrix, test_matrix = builder()
#print("Train Matrix:")
#print(train_matrix)
#print("\nTest Matrix:")
#print(test_matrix)
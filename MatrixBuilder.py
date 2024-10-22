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

    def filter_items(self, data, min_ratings=5):
    
        item_counts = data[self.item_col].value_counts()  # Count ratings per item
        filtered_items = item_counts[item_counts >= min_ratings].index  # Get items that meet the threshold
        filtered_data = data[data[self.item_col].isin(filtered_items)]  # Filter data
        return filtered_data

    def filter_users(self,data,min_ratings = 5):
        user_counts = data[self.user_col].value_counts()
        filtered_users = user_counts[user_counts >= min_ratings].index
        filtered_data = data[data[self.user_col].isin(filtered_users)]
        return filtered_data

    def build_filtered_matrix(self, data, min_interactions=30):
        # Filter users in data 
        filtered_data = self.filter_users(data, min_interactions).copy()
        filtered_data.loc[:, self.user_col] = filtered_data[self.user_col].astype(str)
        filtered_data.loc[:, self.item_col] = filtered_data[self.item_col].astype(str)

        # Get unique users and items
        users = filtered_data[self.user_col].unique()
        items = filtered_data[self.item_col].unique()

        # Create dictionaries to map user and item IDs to indices
        user_to_index = {user: idx for idx, user in enumerate(users)}
        item_to_index = {item: idx for idx, item in enumerate(items)}

        # Initialize the matrix
        user_item_matrix = np.full((len(items), len(users)),np.nan)

        # Fill the matrix with ratings
        for _, row in filtered_data.iterrows():
            user_idx = user_to_index[row[self.user_col]]
            item_idx = item_to_index[row[self.item_col]]
            user_item_matrix[item_idx, user_idx] = row[self.rating_col]
        
        # Convert to a DataFrame for better manipulation
        matrix_df = pd.DataFrame(user_item_matrix, index=items, columns=users)
        matrix_df.index.name = 'ItemId'
        matrix_df.columns.name = 'UserId'
        return matrix_df
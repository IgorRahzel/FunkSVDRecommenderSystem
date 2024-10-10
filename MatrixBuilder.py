import pandas as pd
#from sklearn.model_selection import train_test_split

class MatrixBuilder:
    def __init__(self, user_col='UserId', item_col='ItemId', rating_col='Rating'):
        self.user_col = user_col
        self.item_col = item_col
        self.rating_col = rating_col
    '''
    def split_data(self, data, test_size=0.2, random_state=None):
        train_data, test_data = train_test_split(
            data,
            test_size=test_size,
            random_state=random_state
        )
        return train_data, test_data
    '''
    def filter_users(self, data, min_interactions=5):
        """
        Filters users based on a minimum number of interactions.

        Parameters:
        data (pd.DataFrame): The dataset containing user, item, and rating columns.
        min_interactions (int): Minimum number of interactions a user must have to be kept.

        Returns:
        pd.DataFrame: Filtered dataset with users who have at least `min_interactions`.
        """
        user_counts = data[self.user_col].value_counts()
        filtered_users = user_counts[user_counts >= min_interactions].index
        filtered_data = data[data[self.user_col].isin(filtered_users)]
        return filtered_data

    def build_matrix(self, data):
        user_item_matrix = data.pivot_table(
            index=self.user_col,
            columns=self.item_col,
            values=self.rating_col,
            fill_value=0
        )
        return user_item_matrix

    def build_train_test_matrices(self, data, min_interactions=5):
        """
        Splits the data into training and testing sets, filters users based on minimum interactions,
        and builds user-item matrices for each.

        Parameters:
        data (pd.DataFrame): The dataset containing user, item, and rating columns.
        min_interactions (int): Minimum number of interactions a user must have to be kept.
        test_size (float): Proportion of the dataset to include in the test split.
        random_state (int): Seed used by the random number generator.

        Returns:
        tuple: Training and testing user-item matrices (pd.DataFrame, pd.DataFrame)
        """
        # Filter users before splitting
        filtered_data = self.filter_users(data, min_interactions)
        #train_data, test_data = self.split_data(filtered_data, test_size, random_state)
        
        # Build matrices
        train_matrix = self.build_matrix(filtered_data)
        #test_matrix = self.build_matrix(test_data)
        
        return train_matrix


# Example data
data = pd.DataFrame({
    'UserId': [1, 1, 2, 2, 3, 3, 4, 5, 6],
    'ItemId': ['A', 'B', 'A', 'C', 'B', 'C', 'A', 'B', 'C'],
    'Rating': [5, 3, 4, 2, 1, 5, 4, 3, 2]
})

print(data)

# Instantiate the MatrixBuilder
builder = MatrixBuilder()

# Build train-test matrices, filtering out users with fewer than 2 interactions
train_matrix = builder.build_train_test_matrices(data, min_interactions=2)

print("Train Matrix:")
print(train_matrix)

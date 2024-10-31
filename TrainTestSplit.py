import numpy as np
import pandas as pd

class TrainTestSplit:
    def __init__(self, user_col='UserId', item_col='ItemId', rating_col='Rating', file_name='ratings.csv'):
        self.user_col = user_col        # Name of the user column
        self.item_col = item_col        # Name of the item column
        self.rating_col = rating_col    # Name of the rating column
        self.file = file_name           # File name containing the data
        self.df = None                  # DataFrame created after reading the .csv
    
    def _preProcessing(self):
        # Load data and split 'UserId:ItemId' into separate columns
        df = pd.read_csv(self.file)
        df[['UserId', 'ItemId']] = df['UserId:ItemId'].str.split(':', expand=True)
        df = df.drop(columns=['UserId:ItemId'])  # Drop the original combined column
        self.df = df  # Store the processed DataFrame

    def _splitData(self, test_size=0.2, random_state=None):
        if random_state:
            np.random.seed(random_state)  # Set seed for reproducibility
        random_values = np.random.uniform(0, 1, size=len(self.df))  # Random values for split
        threshold = 1 - test_size  # Threshold for train-test split
        train_data = self.df[random_values < threshold]  # Training data
        test_data = self.df[random_values >= threshold]  # Test data
        return train_data, test_data, self.df  # Return split data and full DataFrame
    
    def __call__(self, test_size=0.2, random_state=None):
        self._preProcessing()  # Preprocess the data
        return self._splitData(test_size=test_size, random_state=random_state)  # Return split data
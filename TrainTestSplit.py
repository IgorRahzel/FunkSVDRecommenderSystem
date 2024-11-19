import numpy as np
import pandas as pd

class TrainTestSplit:
    def __init__(self, user_col='UserId', item_col='ItemId', rating_col='Rating', file_name='ratings.csv'):
        self.user_col = user_col
        self.item_col = item_col
        self.rating_col = rating_col
        self.file = file_name
        self.df = None
    
    def _preProcessing(self):
        df = pd.read_csv(self.file)
        df[['UserId', 'ItemId']] = df['UserId:ItemId'].str.split(':', expand=True)
        df = df.drop(columns=['UserId:ItemId'])
        self.df = df

    def _splitData(self,test_size=0.2, random_state=None):
        if random_state:
            np.random.seed(random_state)
        # Generate a random array of the same length as the DataFrame
        random_values = np.random.uniform(0, 1, size=len(self.df))
        threshold = 1 - test_size
        # Split the DataFrame into training and test sets based on the threshold
        train_data = self.df[random_values < threshold]
        test_data = self.df[random_values >= threshold]
        return train_data, test_data, self.df
    
    def __call__(self,test_size = 0.2,random_state = None):
        self._preProcessing()
        return self._splitData(test_size=test_size,random_state=random_state)
    
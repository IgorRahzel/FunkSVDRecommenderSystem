import pandas as pd
import numpy as np
import sys
from TrainTestSplit import TrainTestSplit
from FunkSVD import FunkSVD


# Helper function to calculate Root Mean Squared Error (RMSE)
def rmse(predictions, actuals):
    return np.sqrt(np.mean((predictions - actuals) ** 2))

#Reading file names
ratings_file = sys.argv[1]
targets_file = sys.argv[2]

# Defining hyperparameters
epochs = 20
lr = 0.1
test_size = 0.001
k = 100
batch_size = 64
lamda = 0.2
    
# Split the data
splitter = TrainTestSplit(file_name=ratings_file)
train_data, test_data, data = splitter(test_size=test_size)
    
# Train the model
model = FunkSVD(train_data)
model.train(k=k, batch_size=batch_size, lr=lr, lamda=lamda, epochs=epochs)
    
# Auxiliary Dataframe of average rating of each item
item_mean = data.groupby('ItemId')['Rating'].mean()
    
# TEST PREDICTION ON TEST SET
'''
test_predictions = []
test_actuals = []

for index, row in test_data.iterrows():
    user = row['UserId']
    item = row['ItemId']
    actual_rating = row['Rating']
    predicted_rating = model.prediction(user, item, item_mean)

    test_predictions.append(predicted_rating)
    test_actuals.append(actual_rating)

    # Calculate RMSE for the test set
    test_predictions = np.array(test_predictions)
    test_actuals = np.array(test_actuals)
    test_rmse = rmse(test_predictions, test_actuals)

    print(f"Test Set RMSE: {test_rmse}")
'''

# Creating the targets dataframe
targets = pd.read_csv(targets_file)
targets[['UserId', 'ItemId']] = targets['UserId:ItemId'].str.split(':', expand=True)
targets = targets.drop(columns=['UserId:ItemId'])

   
# Making predictions for every UserId and ItemId in targets
print('UserId:ItemId,Rating')
for _, row in targets.iterrows():
    user = row['UserId']
    item = row['ItemId']
    rating = model.prediction(user,item,item_mean)
    #Writing prediction to stdout
    print(f'{user}:{item},{rating}')
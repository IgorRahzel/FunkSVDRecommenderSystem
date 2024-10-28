import pandas as pd
import numpy as np
from TrainTestSplit import TrainTestSplit
from FunkSVD import FunkSVD
import itertools

# Helper function to calculate Root Mean Squared Error (RMSE)
def rmse(predictions, actuals):
    return np.sqrt(np.mean((predictions - actuals) ** 2))


# Define the range of hyperparameters to try
param_grid = {
    'epochs': [30],
    'lr': [0.1],
    'test_size': [0.05],  # You can adjust these values for different splits
    'k': [50],
    'batch_size': [64],
    'lamda':[0.1],
}

# Create all combinations of the hyperparameters
param_combinations = list(itertools.product(param_grid['epochs'], param_grid['lr'], param_grid['test_size'], param_grid['k'], param_grid['batch_size'],param_grid['lamda']))

best_rmse = float('inf')
best_config = None

# Iterate over all combinations of hyperparameters
for epochs, lr, test_size, k, batch_size,lamda in param_combinations:
    print(f"Testing configuration: epochs={epochs}, lr={lr}, test_size={test_size}, k={k}, batch_size={batch_size}, lamda={lamda}")
    
    # Split the data
    splitter = TrainTestSplit()
    train_data, test_data, data = splitter(test_size=test_size)
    
    # Train the model
    model = FunkSVD(train_data)
    model.train(k=k, batch_size=batch_size, lr=lr, lamda=lamda, epochs=epochs)
    
    # Auxiliary Dataframe of average rating of each item
    item_mean = data.groupby('ItemId')['Rating'].mean()
    
    # TEST PREDICTION ON TEST SET
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

    # Update the best RMSE and configuration
    if test_rmse < best_rmse:
        best_rmse = test_rmse
        best_config = {
            'epochs': epochs,
            'lr': lr,
            'test_size': test_size,
            'k': k,
            'batch_size': batch_size,
            'lamda': lamda
        }

# Output the best configuration and RMSE
print("\nBest configuration:")
print(best_config)
print(f"Best Test Set RMSE: {best_rmse}")


# Making predictions for the target.csv file

targets = pd.read_csv('targets.csv')
targets[['UserId', 'ItemId']] = targets['UserId:ItemId'].str.split(':', expand=True)
targets = targets.drop(columns=['UserId:ItemId'])

# Abre o arquivo output.csv para escrita
with open('output2.csv', 'w') as file:
    # Escreve o cabeçalho
    file.write('UserId:ItemId,Rating\n')

    # Itera pelas linhas do df_target e escreve cada linha no CSV
    for _, row in targets.iterrows():
        user = row['UserId']
        item = row['ItemId']
        rating = model.prediction(user,item,item_mean)
        # Escreve a linha no arquivo
        file.write(f'{user}:{item},{rating}\n')

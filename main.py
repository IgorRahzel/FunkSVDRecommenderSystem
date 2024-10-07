import pandas as pd
import numpy as np

#print(pd.__version__)
#print(np.__version__)

#Reading the csv file
df = pd.read_csv('ratings.csv')

#Splitting UserID and ItemID into different columns
df[['UserId', 'ItemId']] = df['UserId:ItemId'].str.split(':', expand=True)

#Removing the UserId:ItemId column
df = df.drop(columns=['UserId:ItemId'])

# Set a random seed for reproducibility
np.random.seed(13)

# Generate a random array of the same length as the DataFrame
random_values = np.random.uniform(0, 1, size=len(df))

# Define a threshold for splitting (e.g., 70% for training)
threshold = 0.7

# Split the DataFrame into training and test sets based on the threshold
train_df = df[random_values < threshold]
test_df = df[random_values >= threshold]




#Normalizing the itmes
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

users= train_df['UserId'].unique()

print(train_df)
print(test_df)


# Count the number of ratings per user in the training set
user_rating_count_train = train_df.groupby('UserId').size()

# Get the users who have rated more than one item in the training set
users_with_multiple_ratings_train = user_rating_count_train[user_rating_count_train > 3].index
print(users_with_multiple_ratings_train)

# Filter the training DataFrame to keep only users with multiple ratings
train_df = train_df[train_df['UserId'].isin(users_with_multiple_ratings_train)]

# Creating the user-item matrix for the training set without filling NaNs
user_item_matrix = pd.pivot_table(train_df, values='Rating', index='UserId', columns='ItemId')
print(f"user item matrix1: \n {user_item_matrix}\n\n")

#storing users mean rating
users_mean_rating = user_item_matrix.mean(axis=1,skipna=True)

# Subtract the mean rating per user, ignoring NaNs to only center on rated items
user_item_matrix = user_item_matrix.fillna(0)
user_item_matrix = user_item_matrix.subtract(users_mean_rating, axis=0)
print(f"user item matrix3: \n {user_item_matrix}\n\n")

#implementing cosine similarity
# Identify columns (items) that contain only zeros
non_zero_columns = user_item_matrix.columns[(user_item_matrix != 0).any(axis=0)]

# Filter the user-item matrix to keep only columns with at least one non-zero value
user_item_matrix = user_item_matrix[non_zero_columns]

# Normalize the user-item matrix (L2 normalization for each column)
norms = np.sqrt((user_item_matrix ** 2).sum(axis=0))
normalized_matrix = user_item_matrix / norms

# Fill NaNs that may result from zero divisions (if any norms are zero, though this should be handled above)
normalized_matrix = normalized_matrix.fillna(0)

# Compute the cosine similarity as the dot product of the normalized matrix
cosine_similarity_matrix = normalized_matrix.T.dot(normalized_matrix)

# Convert the result to a DataFrame for easier manipulation and to retain labels
cosine_similarity_df = pd.DataFrame(cosine_similarity_matrix, index=user_item_matrix.columns, columns=user_item_matrix.columns)

print(cosine_similarity_df)

# Assuming you have 'user_item_matrix' and 'cosine_similarity_df' already defined

def predict_rating(user_id, item_id, user_item_matrix, cosine_similarity_df,user_mean):
    # Get the items the user has already rated
    user_ratings = user_item_matrix.loc[user_id]
    rated_items = user_ratings[user_ratings > 0.01].index
    
    # Get similarity scores for the target item with all other items the user has rated
    similarities = cosine_similarity_df[item_id][rated_items]
    
    # Get the user's ratings for those items
    ratings = user_ratings[rated_items]
    
    # Calculate the weighted sum of ratings
    numerator = np.sum(similarities * ratings)
    denominator = np.sum(np.abs(similarities))
    
    # If the denominator is zero (no similar items), return zero
    if denominator == 0:
        return 0
    
    # Calculate the predicted rating
    predicted_rating = numerator / denominator
    return predicted_rating + user_mean[user_id]

# Example usage
users = list(user_item_matrix.index)
flag = True
i = 15
while flag and i < len(users):
    user_id = users[i]
    print(user_item_matrix.loc[user_id])
    # Find an item the user hasn't rated yet
    unrated_items = user_item_matrix.columns[user_item_matrix.loc[user_id] <= 0.01]
    if len(unrated_items) > 0:
        item_id = unrated_items[0]  # Take the first unrated item for this example
        predicted_rating = predict_rating(user_id, item_id, user_item_matrix, cosine_similarity_df,users_mean_rating)
        print(f"Predicted rating for {user_id} on {item_id}: {predicted_rating}")
        flag = False
    else:
        print(f"User {user_id} has rated all items.")
        i=i+1

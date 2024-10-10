import pandas as pd
import numpy as np

# Assuming 'user_item_matrix' is already defined as a Pandas DataFrame from your previous steps
import pandas as pd

# Creating a small user-item matrix DataFrame
data = {
    'Item1': [5, 0, 3],
    'Item2': [4, 0, None],
    'Item3': [None, 2, 4],
    'Item4': [0, None, 1],
}

# Each row represents a user, and each column represents an item
user_item_matrix = pd.DataFrame(data, index=['User1', 'User2', 'User3'])
user_mean = user_item_matrix.mean(axis=1,skipna=True)
normalized_matrix = user_item_matrix - user_mean
print("User-Item Matrix:")
print(f"{user_item_matrix}\n\n")
print(f"mean:\n {user_mean}")
print(f"normalized matrix:\n {normalized_matrix}")

'''
# Normalize the user-item matrix (L2 normalization for each column)
norms = np.sqrt((user_item_matrix ** 2).sum(axis=0))
print(f"Norms:\n {norms} \n\n")
normalized_matrix = user_item_matrix / norms
print(f"normalized matrix:\n {normalized_matrix} \n\n")

# Compute the cosine similarity as the dot product of the normalized matrix
cosine_similarity_matrix = normalized_matrix.T.dot(normalized_matrix)


# Convert the result to a DataFrame for easier manipulation and to retain labels
cosine_similarity_df = pd.DataFrame(cosine_similarity_matrix, index=user_item_matrix.columns, columns=user_item_matrix.columns)

print(cosine_similarity_df)
'''
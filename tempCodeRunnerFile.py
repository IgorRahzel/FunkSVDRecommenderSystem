# Normalize the user-item matrix (L2 normalization for each column)
norms = np.sqrt((user_item_matrix ** 2).sum(axis=0))
normalized_matrix = user_item_matrix / norms

# Fill NaNs that may result from zero divisions (if any norms are zero, though this should be handled above)
normalized_matrix = normalized_matrix.fillna(0)

# Compute the cosine similarity as the dot product of the normalized matrix
cosine_similarity_matrix = normalized_matrix.T.dot(normalized_matrix)

# Convert the result to a DataFrame for easier manipulation and to retain labels
cosine_similarity_df = pd.DataFrame(cosine_similarity_matrix, index=user_item_matrix.columns, columns=user_item_matrix.columns)
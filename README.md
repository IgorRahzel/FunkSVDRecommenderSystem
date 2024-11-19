# FunkSVD Recommender System

This repository implements a recommendation system using **FunkSVD (Funk Singular Value Decomposition)**. It predicts user-item ratings based on collaborative filtering, leveraging techniques like gradient descent optimization and bias adjustment.

## Features
- **FunkSVD Algorithm**: A matrix factorization-based method for collaborative filtering.
- **Hyperparameter Tuning**: Includes options for tuning learning rate, regularization, and other key parameters.
- **Mini-batch Gradient Descent**: Efficient training with support for Adam optimizer.
- **Train-Test Split**: Custom utility to split data into training and testing sets.
- **Prediction**: Generates predictions for user-item pairs, including support for missing data.

---

## File Structure

### **1. `main.py`**
- The entry point for the system.
- Automates hyperparameter tuning using grid search.
- Splits data into train and test sets, trains the model, and evaluates performance using RMSE.
- Generates predictions for user-item pairs in `targets.csv` and saves results to `output3.csv`.

### **2. `FunkSVD.py`**
- Contains the implementation of the FunkSVD algorithm:
  - Handles matrix factorization (`P` and `Q` matrices).
  - Supports bias adjustments for users and items.
  - Includes mini-batch gradient descent and the Adam optimizer for parameter updates.

### **3. `TrainTestSplit.py`**
- Handles preprocessing and splitting of the dataset into training and testing sets.
- Reads user-item rating data from `ratings.csv`.

### **4. `Adam.py`**
- Implements the **Adam Optimizer**, a popular optimization algorithm used in training machine learning models.
- Provides efficient updates for parameters during mini-batch gradient descent.

---

## How to Use

### **1. Prerequisites**
Ensure you have the following dependencies installed:
```bash
pip install numpy pandas
```
### **2. Prepare the Data**
Place a `ratings.csv` file in the project directory. The file should have the format:

UserId:ItemId,Rating

### **3. Run the Code**
Execute `main.py` to:
1. Tune hyperparameters.
2. Train the FunkSVD model.
3. Generate predictions.

```bash
python3 main.py
```

### **4. Output**
Predictions are saved to `outuput.csv`

### **5. Hyperparameters**
Key Hyperparameters for tuning the model
- `epochs`: Number of training iterations.
- `lr`: Learning rate for gradient descent.
- `k`: Number of latent factors.
- `batch_size`: Size of mini-batches.
- `lamda`: Regularization strength.
- `test_size`: Fraction of data used for testing.

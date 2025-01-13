import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
import time

# Start timer
start_time = time.time()

# Load the insurance dataset
df = pd.read_csv('../trainingData/insurance.csv')
print("First 5 rows of the dataset:")
print(df.head())

# Check for missing values
print("\nMissing values in each column:")
print(df.isnull().sum())

# Encode categorical variables

# Encode 'sex' column: 'female' -> 0, 'male' -> 1
df['sex'] = df['sex'].map({'female': 0, 'male': 1}).astype(int)

# Encode 'smoker' column: 'no' -> 0, 'yes' -> 1
df['smoker'] = df['smoker'].map({'no': 0, 'yes': 1}).astype(int)

# One-hot encode 'region' column
region_dummies = pd.get_dummies(df['region'], prefix='region')
df = pd.concat([df, region_dummies], axis=1)
df = df.drop('region', axis=1)

# Convert boolean dummy columns to integers
region_columns = [col for col in df.columns if col.startswith('region_')]
df[region_columns] = df[region_columns].astype(int)

# Verify the conversion
print("\nData Types After Encoding and Conversion:")
print(df.dtypes)

# Handle missing values in numerical columns by replacing with mean
numerical_cols = ['age', 'bmi', 'children']
for col in numerical_cols:
    if df[col].isnull().sum() > 0:
        df[col] = df[col].fillna(df[col].mean())

# Normalize numerical features
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
df[numerical_cols] = scaler.fit_transform(df[numerical_cols])

# Optional: Normalize the target variable 'charges' if necessary
scaler_target = StandardScaler()
df['charges'] = scaler_target.fit_transform(df[['charges']])

# Separate features and target
X = df.drop('charges', axis=1).values
y = df['charges'].values.reshape(-1, 1)  # Ensure y is of shape (n_samples, 1)

print(df.head());

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Convert to PyTorch tensors
X_train = torch.tensor(X_train, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.float32)
X_test = torch.tensor(X_test, dtype=torch.float32)
y_test = torch.tensor(y_test, dtype=torch.float32)

# Define regression metrics
def mean_absolute_error(y_pred, y_true):
    return torch.mean(torch.abs(y_pred - y_true))

def root_mean_squared_error(y_pred, y_true):
    return torch.sqrt(torch.mean((y_pred - y_true) ** 2))

# Define the Neural Network for Regression
class InsuranceNetwork(nn.Module):
    def __init__(self, input_size):
        super(InsuranceNetwork, self).__init__()
        torch.manual_seed(0)
        # self.net = nn.Sequential(
        #     nn.Linear(input_size, 128),
        #     nn.ReLU(),
        #     nn.Linear(128, 256),
        #     nn.ReLU(),
        #     nn.Linear(256, 64),
        #     nn.ReLU(),
        #     nn.Linear(64, 1)  # Single output for regression
        # )
        self.net = nn.Sequential(
             nn.Linear(input_size, 6),  # Input to Hidden Layer
             nn.ReLU(),                           # Activation Function
             nn.Linear(6, 1)            # Hidden to Output Layer
         )

    def forward(self, x):
        return self.net(x)

    def predict(self, x):
        return self.forward(x)

# Initialize the model
input_size = X_train.shape[1]
model = InsuranceNetwork(input_size)

# Choose device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
X_train = X_train.to(device)
y_train = y_train.to(device)
X_test = X_test.to(device)
y_test = y_test.to(device)

# Define loss function and optimizer
loss_fn = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training function
def train(model, optimizer, loss_fn, X, y, epochs=1000):
    model.train()
    for epoch in range(epochs):
        # Forward pass
        y_pred = model(X)
        loss = loss_fn(y_pred, y)

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Print loss every 100 epochs
        if (epoch+1) % 100 == 0:
            print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}")
    return loss.item()

# Train the model
final_loss = train(model, optimizer, loss_fn, X_train, y_train, epochs=1000)
print(f"\nFinal Training Loss: {final_loss:.4f}")

# Evaluation
model.eval()
with torch.no_grad():
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)

    train_mae = mean_absolute_error(y_pred_train, y_train).item()
    train_rmse = root_mean_squared_error(y_pred_train, y_train).item()

    test_mae = mean_absolute_error(y_pred_test, y_test).item()
    test_rmse = root_mean_squared_error(y_pred_test, y_test).item()

print(f"\nTraining MAE: {train_mae:.2f}")
print(f"Training RMSE: {train_rmse:.2f}")
print(f"Testing MAE: {test_mae:.2f}")
print(f"Testing RMSE: {test_rmse:.2f}")

# End timer
end_time = time.time()
elapsed_time = end_time - start_time
print(f"\nElapsed time: {elapsed_time:.4f} seconds")

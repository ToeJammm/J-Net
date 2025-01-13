import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import torch
import math
import matplotlib.pyplot as plt
import matplotlib.colors
import torch.nn.functional as F
from optparse import Values
from jedi.inference.base_value import ValueSet
from typing import ValuesView
import torch.nn as nn
from torch import optim
import time

start_time = time.time()


df_train = pd.read_csv('../trainingData/train.csv')
 print(df_train.head(5))

# We can't do anything with the Name, Ticket number, and Cabin, so we drop them
df_train = df_train.drop(['PassengerId','Name','Ticket', 'Cabin'], axis=1)

# replace 'female' by 0 and 'male' by 1
df_train['Sex'] = df_train['Sex'].map({'female':0, 'male':1}).astype(int)

# Create dummy variables
embarked_dummies = pd.get_dummies(df_train['Embarked'], prefix='Embarked').astype(int)

# Concatenate with original DataFrame
df_train = pd.concat([df_train, embarked_dummies], axis=1)

# Drop the original 'Embarked' column
df_train = df_train.drop('Embarked', axis=1)


# We normalize the age and the fare
age_mean = df_train['Age'].mean()
age_std = df_train['Age'].std()
df_train['Age'] = (df_train['Age'] - age_mean) / age_std

fare_mean = df_train['Fare'].mean()
fare_std = df_train['Fare'].std()
df_train['Fare'] = (df_train['Fare'] - fare_mean) / fare_std

# In many cases, the 'Age' is missing - which can cause problems

# handle these missing values by replacing them by the mean age.
df_train['Age'] = df_train['Age'].fillna(df_train['Age'].mean())

# print(df_train.head())

#separate target column
X_train = df_train.drop('Survived', axis=1).to_numpy()
y_train = df_train['Survived'].to_numpy()
X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=0.2)



X_train, y_train, X_test, y_test = map(torch.tensor, (X_train, y_train, X_test, y_test))


# print(X_train.shape, y_train.shape)

def accuracy(y_hat, y):
  pred = torch.argmax(y_hat, dim=1)
  return (pred == y).float().mean()



X_train = X_train.float()
y_train = y_train.long()

X_test = X_test.float()
y_test = y_test.long()


##create NN class
class FirstNetwork_v3(nn.Module):

  def __init__(self):
    super().__init__()
    torch.manual_seed(0)
    self.net = nn.Sequential(
        nn.Linear(9, 128),
        nn.ReLU(),
        nn.Linear(128, 256),
        nn.ReLU(),
        nn.Linear(256, 2),
        nn.Softmax()
    )

  def forward(self, X):
    return self.net(X)

  def predict(self, X):
    Y_pred = self.forward(X)
    return Y_pred

def fit_v2(x, y, model, opt, loss_fn, epochs = 1000):

  for epoch in range(epochs):
    loss = loss_fn(model(x), y)
    loss.backward()
    opt.step()
    opt.zero_grad()

  return loss.item()



device = torch.device("mps")

X_train=X_train.to(device)
y_train=y_train.to(device)
X_test=X_test.to(device)
y_test=y_test.to(device)
fn = FirstNetwork_v3()
fn.to(device)
loss_fn = F.cross_entropy
opt = optim.SGD(fn.parameters(), lr=0.5)

print('Final loss', fit_v2(X_train, y_train, fn, opt, loss_fn))


Y_pred_train = fn.predict(X_train)
#Y_pred_train = np.argmax(Y_pred_train,1)
Y_pred_val = fn.predict(X_test)
#Y_pred_val = np.argmax(Y_pred_val,1)

accuracy_train = accuracy(Y_pred_train, y_train)
accuracy_val = accuracy(Y_pred_val, y_test)

end_time = time.time()

print("Training accuracy", (accuracy_train))
print("Validation accuracy",(accuracy_val))

elapsed_time = end_time - start_time
print(f"Elapsed time: {elapsed_time:.4f} seconds")

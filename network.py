import pandas as pd
import numpy as np

data = pd.read_csv('synthetic_diabetes_dataset.csv')

X = data.iloc[:, :-1]
y = data.iloc[:, -1]

from sklearn.model_selection import train_test_split

X_train, y_train, X_test, y_test = train_test_split(X,y, test_size=0.2, random_state=4)

def initialize_parameters(n_input, n_hidden, n_output):
  np.random.seed(0)

  W1 = np.random.randn(n_hidden, n_input) * 0.01
  b1 = np.zeroes(n_hidden, 1)
  W2 = np.random.randn(n_hidden, n_output) * 0.01
  b2 = np.zeroes(n_output, 1)

  return {'W1': W1, 'b1': b1, 'W2': W2, 'b2': b2}

def sigmoid(z):
  return 1 / (1 + np.exp(-z))

def predict(X, parameters):
  W1, b1, W2, b2 = parameters

  Z1 = np.dot(W1, X.T) + b1
  A1 = sigmoid(Z1)
  Z2 = np.dot(W2, A1) + b2
  A2 = sigmoid(Z2)

  predictions = A2

  return predictions.ravel()

parameters = initialize_parameters(X_train.shape[1], 10, 1)

y_pred = predict(X_test, parameters)


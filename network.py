import pandas as pd
import numpy as np

data = pd.read_csv('synthetic_diabetes_dataset.csv')

X = data.iloc[:, :-1]
y = data.iloc[:, -1]

from sklearn.model_selection import train_test_split

X_train, y_train, X_test, y_test = train_test_split(X,y, test_size=0.2, random_state=40)


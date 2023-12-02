import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

data = pd.read_csv('./data/clean-data.csv')

X = np.array(data['Questions']).astype('str')
y = np.array(data['Answers']).astype('str')

print('Splitting dataset to train/test sets with ratio of 70-30')
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7, random_state=False)

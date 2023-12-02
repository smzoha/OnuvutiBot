import numpy as np
import pandas as pd
from bnlp.tokenizer.basic import BasicTokenizer
from sklearn.model_selection import train_test_split


def get_tokens(X, y):
    tokens = []
    tokenizer = BasicTokenizer()

    for sentence in X:
        tokens.extend(tokenizer.tokenize(sentence))

    for sentence in y:
        tokens.extend(tokenizer.tokenize(sentence))

    return list(set(tokens))


data = pd.read_csv('./data/clean-data.csv')

X = np.array(data['Questions']).astype('str')
y = np.array(data['Answers']).astype('str')

print('Splitting dataset to train/test sets with ratio of 70-30')
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7, random_state=False)

tokens = get_tokens(X, y)
print(tokens)

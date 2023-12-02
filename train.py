import numpy as np
import pandas as pd
from bnlp.embedding.word2vec import BengaliWord2Vec
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


def word_embeddings(tokens, verbose):
    glove = BengaliWord2Vec()
    vectors = []
    failed_tokens = []

    for token in tokens:
        try:
            vectors.append(glove.get_word_vector(token))
        except KeyError:
            failed_tokens.append(token)

            if verbose:
                print('Failed to generate for token:', token)

    return vectors, failed_tokens


data = pd.read_csv('./data/clean-data.csv', encoding='utf-8-sig')

X = np.array(data['Questions']).astype('str')
y = np.array(data['Answers']).astype('str')

print('Splitting dataset to train/test sets with ratio of 70-30')
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7, random_state=False)

tokens = get_tokens(X, y)
print('Generated tokens. Vocabulary count:', len(tokens))

vectors, failed_tokens = word_embeddings(tokens, verbose=False)
print('Generated Word Embeddings from Tokens. Vector Count:', len(vectors))
print('Failed Token Count:', len(failed_tokens))

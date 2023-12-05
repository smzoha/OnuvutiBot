import numpy as np
import pandas as pd
from bnlp.embedding.word2vec import BengaliWord2Vec
from bnlp.tokenizer.basic import BasicTokenizer
from sklearn.model_selection import train_test_split


def get_vocab(X, y):
    tokens = []
    tokenizer = BasicTokenizer()

    for sentence in X:
        tokens.extend(tokenizer.tokenize(sentence))

    for sentence in y:
        tokens.extend(tokenizer.tokenize(sentence))

    return list(set(tokens))


def word_embeddings(vocab, verbose):
    word2vec = BengaliWord2Vec()
    vectors = []
    failed_tokens = []

    for word in vocab:
        try:
            vectors.append(word2vec.get_word_vector(word))
        except KeyError:
            vectors.append(np.zeros(100))
            failed_tokens.append(word)

            if verbose:
                print('Failed to generate for token:', word)

    return vectors, failed_tokens


data = pd.read_csv('./data/clean-data.csv', encoding='utf-8-sig')

X = np.array(data['Questions']).astype('str')
y = np.array(data['Answers']).astype('str')

vocab = get_vocab(X, y)
print('Generated vocab. Vocabulary count:', len(vocab))

embeddings, failed_words = word_embeddings(vocab, verbose=False)
print('Generated Word Embeddings from Tokens. Vector Count:', len(embeddings))
print('Failed Token Count (patched with zeroes):', len(failed_words))

print('Splitting dataset to train/test sets with ratio of 70-30')
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7, random_state=False)

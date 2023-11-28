import os
import string

import pandas as pd


def preprocess(unprocessed_data, attr):
    unprocessed_data[attr] = unprocessed_data[attr].str.replace('[^\\w\\s]', '', regex=True)
    unprocessed_data[attr] = unprocessed_data[attr].str.replace('[\\s]{2,}', '', regex=True)

    return unprocessed_data


# Load dataset
data = pd.read_csv('./data/BengaliEmpatheticConversationsCorpus.csv')
data = data[['Questions', 'Answers']]

# Print dataset summary
print('Data summary (before preprocessing)')
print(data.describe())

print('===================================')

# Drop null values and duplicates, and print dataset summary
data = data.dropna()
data = data.drop_duplicates()

print('Data summary (after dropping na/duplicates)')
print(data.describe())

print('===================================')

# Print first 10 Questions/Answers
print('First 10 Questions/Answers')
print(data.head(10))

print('===================================')

# Cleanup unwanted characters
data = preprocess(data, 'Questions')
data = preprocess(data, 'Answers')

print('First 10 Questions/Answers (after processing)')
print(data.head(10))

print('===================================')

print('Writing cleaned-up data to file')
if not os.path.exists('./data'):
    os.mkdir('./data')

data.to_csv('./data/clean-data.csv', index=False)
print('Write successful to `./data/clean-data.csv`')

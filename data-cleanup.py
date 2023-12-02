import os

import bnlp
import pandas as pd


def preprocess(input_data):
    cleaner = bnlp.CleanText(remove_url=True, remove_email=True)

    for i in range(0, len(input_data)):
        tmp = input_data[i]
        tmp = tmp.strip()
        input_data[i] = cleaner(tmp)

    return input_data


# Load dataset
data = pd.read_csv('./data/BengaliEmpatheticConversationsCorpus .csv')
data = data[['Questions', 'Answers']]

# Print dataset summary
print('Data summary (before preprocessing)')
print(data.describe())

print('===================================')

# Drop null values and duplicates, and print dataset summary
data = data.dropna()
data = data.drop_duplicates()
data = data.reset_index(drop=True)

print('Data summary (after dropping na/duplicates)')
print(data.describe())

print('===================================')

# Print first 10 Questions/Answers
print('First 10 Questions/Answers')
print(data.head(10))

print('===================================')

# Cleanup unwanted characters
data['Questions'] = preprocess(data['Questions'])
data['Answers'] = preprocess(data['Answers'])

print('First 10 Questions/Answers (after processing)')
print(data.head(10))

print('===================================')

print('Data summary (after cleaning up)')
print(data.describe())
print('===================================')

print('Writing cleaned-up data to file')
if not os.path.exists('./data'):
    os.mkdir('./data')

data.to_csv('./data/clean-data.csv', index=False, encoding='utf-8-sig')
print('Write successful to `./data/clean-data.csv`')

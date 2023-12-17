import os

import pandas as pd

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

print('Writing cleaned-up data to file')
if not os.path.exists('./data'):
    os.mkdir('./data')

data.to_csv('./data/clean-data.csv', index=False, encoding='utf-8-sig')
print('Write successful to `./data/clean-data.csv`')

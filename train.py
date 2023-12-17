import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer

data = pd.read_csv('data/clean-data.csv', encoding='utf-8-sig')

x = data['Questions']
y = data['Answers']

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3)

tokenizer = Tokenizer()
tokenizer.fit_on_texts(x_train)
tokenizer.fit_on_texts(y_train)

vocab_size = len(tokenizer.word_index) + 1
print(vocab_size)

x_train_token = tokenizer.texts_to_sequences(x_train)
y_train_token = tokenizer.texts_to_sequences(y_train)

x_train_seq = pad_sequences(x, maxlen=512, padding='post', truncating='post')
y_train_seq = pad_sequences(y, maxlen=512, padding='post', truncating='post')

model = Sequential(
    Embedding(vocab_size, 256, input_length=x_train_seq.shape[1]),
    LSTM(256, return_sequences=True),
    Dense(vocab_size, activation='softmax')
)

model.compile(optimizer='adam', loss='spatial_categorical_crossentropy',
              metrics=['accuracy', 'precision', 'recall'])

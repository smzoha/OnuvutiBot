import pandas as pd
from keras import Sequential
from keras.src.layers import LSTM, Embedding, Dense
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

data = pd.read_csv('data/clean-data.csv', encoding='utf-8-sig')

x = data['Questions'].astype('str').values
y = data['Answers'].astype('str').values

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3)

tokenizer = Tokenizer()
tokenizer.fit_on_texts(x_train)
tokenizer.fit_on_texts(y_train)

vocab_size = len(tokenizer.word_index) + 1

x_train_seq = tokenizer.texts_to_sequences(x_train)
y_train_seq = tokenizer.texts_to_sequences(y_train)
max_seq_length = max(max([len(seq) for seq in x_train_seq]), max([len(seq) for seq in y_train_seq]))

x_train = pad_sequences(x_train_seq, maxlen=max_seq_length, padding='post', truncating='post')
y_train = pad_sequences(y_train_seq, maxlen=max_seq_length, padding='post', truncating='post')

model = Sequential([
    Embedding(vocab_size, 64, input_length=max_seq_length, mask_zero=True),
    LSTM(64, return_sequences=True),
    Dense(vocab_size, activation='softmax')
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.fit(x_train, y_train, epochs=50, verbose=1)

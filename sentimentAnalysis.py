# Import necessary libraries
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
import numpy as np

# Sample data (replace with your dataset)
texts = ["This is a positive sentence.", "Negative sentiment in this one.", "Neutral statement here."]

labels = [1, 0, 2]  # 1 for positive, 0 for negative, 2 for neutral

# Tokenize and pad sequences
tokenizer = Tokenizer()
tokenizer.fit_on_texts(texts)
total_words = len(tokenizer.word_index) + 1

sequences = tokenizer.texts_to_sequences(texts)
padded_sequences = pad_sequences(sequences, maxlen=max(len(s) for s in sequences), padding='post')

# Convert labels to one-hot encoding
labels_one_hot = tf.keras.utils.to_categorical(labels, num_classes=3)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(padded_sequences, labels_one_hot, test_size=0.2, random_state=42)

# Build the LSTM model
model = Sequential()
model.add(Embedding(total_words, 32, input_length=max(len(s) for s in sequences)))
model.add(LSTM(100))
model.add(Dense(3, activation='softmax'))  # 3 classes for positive, negative, neutral

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train, epochs=5, validation_data=(X_test, y_test))

# Save the model
model.save('sentiment_analysis_model.h5')

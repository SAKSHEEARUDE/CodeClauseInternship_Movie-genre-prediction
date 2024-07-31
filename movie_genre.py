import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Embedding, LSTM
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split

# Sample movie data
data = {
    'plot': [
        'A young wizard discovers his magical heritage and battles a dark wizard.',
        'A group of friends embark on a quest to destroy a powerful ring.',
        'A lawyer defends a young boy accused of murder.',
        'An astronaut becomes stranded on Mars and must find a way to survive.',
        'A superhero fights crime in a city plagued by corruption.',
        'A romantic comedy about two people who meet in a cafe and fall in love.',
        'A horror film where a family encounters paranormal activities in their new home.'
    ],
    'action': [1, 1, 0, 0, 1, 0, 0],
    'adventure': [1, 1, 0, 1, 1, 0, 0],
    'drama': [0, 0, 1, 0, 0, 1, 0],
    'comedy': [0, 0, 0, 0, 0, 1, 0],
    'horror': [0, 0, 0, 0, 0, 0, 1]
}

# Create DataFrame
df = pd.DataFrame(data)

# Features and labels
X = df['plot']
y = df[['action', 'adventure', 'drama', 'comedy', 'horror']]

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Tokenization and padding
tokenizer = Tokenizer(num_words=10000)
tokenizer.fit_on_texts(X_train)
X_train_seq = tokenizer.texts_to_sequences(X_train)
X_test_seq = tokenizer.texts_to_sequences(X_test)
X_train_padded = pad_sequences(X_train_seq, maxlen=100)
X_test_padded = pad_sequences(X_test_seq, maxlen=100)

# Model
model = Sequential()
model.add(Embedding(input_dim=10000, output_dim=128, input_length=100))
model.add(LSTM(128, return_sequences=True))
model.add(LSTM(64))
model.add(Dense(5, activation='sigmoid'))  # 5 genres

# Compile
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train
model.fit(X_train_padded, y_train, epochs=10, batch_size=2, validation_split=0.1)

# Evaluate
loss, accuracy = model.evaluate(X_test_padded, y_test)
print("Test Accuracy:", accuracy)

# Predict function
def predict_genre(plot):
    plot_seq = tokenizer.texts_to_sequences([plot])
    plot_padded = pad_sequences(plot_seq, maxlen=100)
    prediction = model.predict(plot_padded)
    genres = ['Action', 'Adventure', 'Drama', 'Comedy', 'Horror']
    predicted_genres = [genres[i] for i in range(len(prediction[0])) if prediction[0][i] > 0.5]
    return predicted_genres

# Example usage
new_plot = "A young woman faces supernatural challenges after moving into a haunted house."
print("Predicted Genres:", predict_genre(new_plot))

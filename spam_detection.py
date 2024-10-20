import pandas as pd
import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Bidirectional, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping

data = {
    'label': ['spam', 'ham', 'spam', 'ham', 'spam'],
    'message': [
        'Congratulations, you won a lottery!',
        'Hey, how are you doing today?',
        'Claim your free prize now!',
        'Let\'s meet for lunch tomorrow.',
        'Urgent! Your account has been hacked.'
    ]
}
df = pd.DataFrame(data)
df['label'] = df['label'].map({'spam': 1, 'ham': 0})
max_vocab_size = 1000
max_sequence_length = 10
tokenizer = Tokenizer(num_words=max_vocab_size, oov_token='<OOV>')
tokenizer.fit_on_texts(df['message'])
sequences = tokenizer.texts_to_sequences(df['message'])


padded_sequences = pad_sequences(sequences, maxlen=max_sequence_length, padding='post')

X_train, X_test, y_train, y_test = train_test_split(padded_sequences, df['label'], test_size=0.2, random_state=42)

model = Sequential([
    Embedding(max_vocab_size, 64, input_length=max_sequence_length),
    Bidirectional(LSTM(64, return_sequences=True)),
    Bidirectional(LSTM(32)),
    Dropout(0.5),  # Added Dropout to prevent overfitting
    Dense(32, activation='relu'),
    Dense(1, activation='sigmoid')
])

# Compile the model with a lower learning rate
optimizer = Adam(learning_rate=0.001)  # Adjusted learning rate for better convergence
model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])

# Implement EarlyStopping
early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

# Train the model
history = model.fit(X_train, y_train, epochs=20, validation_data=(X_test, y_test), 
                    callbacks=[early_stopping], batch_size=16, verbose=2)

# Evaluate the model
loss, accuracy = model.evaluate(X_test, y_test, verbose=2)
print(f'Test Accuracy: {accuracy:.8f}')

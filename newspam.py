import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report
from keras.models import Sequential
from keras.layers import Dense, LSTM, Embedding, SpatialDropout1D
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer

# Load the dataset
data_path = 'emails.csv/data_split.csv'
df = pd.read_csv(data_path)

# Display initial DataFrame shape
print(f'Initial DataFrame shape: {df.shape}')

# Sample 10% of the data
df_sampled = df.sample(frac=0.1, random_state=42)
print(f'Shape after sampling: {df_sampled.shape}')

# Display unique values in the 'file' column after sampling
print(f'Unique values in \'file\' after sampling: {df_sampled["file"].unique()}')

# Encode the labels
label_encoder = LabelEncoder()
df_sampled['label'] = label_encoder.fit_transform(df_sampled['file'].str.contains('spam'))  # Assuming 'spam' indicates spam emails

# Prepare the text data for LSTM
X = df_sampled['message'].values  # Text data
y = df_sampled['label'].values  # Labels

# Tokenize the text data
tokenizer = Tokenizer()
tokenizer.fit_on_texts(X)
X_sequences = tokenizer.texts_to_sequences(X)
X_padded = pad_sequences(X_sequences)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_padded, y, test_size=0.2, random_state=42)

# Define the LSTM model
model = Sequential()
model.add(Embedding(input_dim=len(tokenizer.word_index) + 1, output_dim=128, input_length=X_padded.shape[1]))
model.add(SpatialDropout1D(0.2))
model.add(LSTM(100, dropout=0.2, recurrent_dropout=0.2))
model.add(Dense(1, activation='sigmoid'))  # For binary classification

# Compile the model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train, epochs=5, batch_size=64, validation_data=(X_test, y_test))

# Evaluate the model
y_pred = (model.predict(X_test) > 0.5).astype("int32")  # Convert probabilities to binary predictions
print(classification_report(y_test, y_pred))

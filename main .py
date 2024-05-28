import json
import re
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import LSTM, Dense, Embedding
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
from preprocess import preprocess_dataset

# Load and preprocess the dataset
json_file = 'intents.json'
input_texts, target_texts, intent_response_map = preprocess_dataset(json_file)

# Define tokenizer for input texts
input_tokenizer = Tokenizer()
input_tokenizer.fit_on_texts(input_texts)

# Convert input text sequences to integer sequences
input_sequences = input_tokenizer.texts_to_sequences(input_texts)

# Pad input sequences to ensure uniform length
max_len = max(len(seq) for seq in input_sequences)
input_sequences = pad_sequences(input_sequences, maxlen=max_len, padding='post')

# Define tokenizer for target texts
target_tokenizer = Tokenizer()
target_tokenizer.fit_on_texts(target_texts)

# Check the tokenizers
print(f"Input Tokenizer Word Index: {input_tokenizer.word_index}")
print(f"Target Tokenizer Word Index: {target_tokenizer.word_index}")

# Convert target text sequences to integer sequences
# Ensure all target texts are in the word index
target_sequences = []
for word in target_texts:
    if word in target_tokenizer.word_index:
        target_sequences.append(target_tokenizer.word_index[word])
    else:
        target_sequences.append(0)  # Use 0 for unknown words

# Ensure that target sequences are numpy arrays
target_sequences = np.array(target_sequences)

# Define vocabulary size for inputs and targets
input_vocab_size = len(input_tokenizer.word_index) + 1
target_vocab_size = len(target_tokenizer.word_index) + 1

# Define the model architecture
model = Sequential([
    Embedding(input_vocab_size, 128, input_length=max_len),
    LSTM(256),
    Dense(target_vocab_size, activation='softmax')
])

# Compile the model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(input_sequences, target_sequences, epochs=10, verbose=1)

# Function to generate response
def generate_response(input_text):
    input_seq = input_tokenizer.texts_to_sequences([input_text])
    padded_input_seq = pad_sequences(input_seq, maxlen=max_len, padding='post')
    predicted_probs = model.predict(padded_input_seq)
    predicted_id = np.argmax(predicted_probs, axis=-1)[0]
    predicted_label = target_tokenizer.index_word.get(predicted_id, '')

    # Debugging logs
    print(f"Input text: {input_text}")
    print(f"Predicted ID: {predicted_id}")
    print(f"Predicted Label: {predicted_label}")

    if predicted_label in intent_response_map:
        responses = intent_response_map[predicted_label]
        return np.random.choice(responses)  # Return a random response from the list of responses
    else:
        return "Sorry, I don't understand."

# Test the chatbot
while True:
    user_input = input("You: ")
    response = generate_response(user_input)
    print("Chatbot:", response)

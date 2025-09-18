import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import SimpleRNN, Dense, Embedding
from tensorflow.keras.preprocessing.sequence import pad_sequences

# 1. Input long paragraph
para = input("Enter a paragraph: ").lower()

# 2. Tokenize paragraph into words
tokenizer = Tokenizer()
tokenizer.fit_on_texts([para])
word_index = tokenizer.word_index
index_word = {v: k for k, v in word_index.items()}
total_words = len(word_index) + 1

# 3. Create word sequences (each input with its next word)
tokens = para.split()
input_sequences = []
for i in range(1, len(tokens)):
    words = tokens[:i+1]
    encoded = tokenizer.texts_to_sequences([' '.join(words)])[0]
    input_sequences.append(encoded)

# 4. Pad all sequences
max_len = max(len(seq) for seq in input_sequences)
input_sequences = pad_sequences(input_sequences, maxlen=max_len)

# 5. Split inputs and labels
X = input_sequences[:, :-1]
y = input_sequences[:, -1]
y = tf.keras.utils.to_categorical(y, num_classes=total_words)

# 6. Build the RNN model
model = Sequential([
    Embedding(input_dim=total_words, output_dim=32, input_length=max_len-1),
    SimpleRNN(64),
    Dense(total_words, activation='softmax')
])
model.compile(loss='categorical_crossentropy', optimizer='adam')

# 7. Train the model
model.fit(X, y, epochs=300, verbose=0)

# 8. Predict next word
seed_text = input("\nEnter starting words: ").lower()
next_words = 10  # you can increase this

for _ in range(next_words):
    token_list = tokenizer.texts_to_sequences([seed_text])[0]
    token_list = pad_sequences([token_list], maxlen=max_len-1)
    predicted = model.predict(token_list, verbose=0)
    predicted_index = np.argmax(predicted)
    predicted_word = index_word.get(predicted_index, '?')
    seed_text += ' ' + predicted_word

print("\nGenerated text:")
print(seed_text)

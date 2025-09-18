import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import SimpleRNN, Dense
from tensorflow.keras.utils import to_categorical

# 1. Take paragraph input
paragraph = input("Enter a paragraph: ").lower()

# 2. Tokenize paragraph into words
tokens = paragraph.split()
tokenizer = Tokenizer()
tokenizer.fit_on_texts([paragraph])
word_index = tokenizer.word_index
index_word = {v: k for k, v in word_index.items()}
vocab_size = len(word_index) + 1

# 3. Create input-output word pairs
X_seq = [word_index[w] for w in tokens[:-1]]
Y_seq = [word_index[w] for w in tokens[1:]]
X = to_categorical(X_seq, num_classes=vocab_size).reshape(1, -1, vocab_size)
Y = to_categorical(Y_seq, num_classes=vocab_size).reshape(1, -1, vocab_size)

# 4. Build RNN model
model = Sequential([
    SimpleRNN(64, return_sequences=True, input_shape=(None, vocab_size)),
    Dense(vocab_size, activation='softmax')
])
model.compile(loss='categorical_crossentropy', optimizer='adam')

# 5. Train the model
model.fit(X, Y, epochs=300, verbose=0)

# 6. Predict next words
output = model.predict(X, verbose=0)
predicted_indices = np.argmax(output, axis=2)[0]
predicted_words = [index_word.get(i, '?') for i in predicted_indices]

# 7. Show results
print("\nOriginal paragraph    :", paragraph)
print("Predicted next words  :", ' '.join(predicted_words))

print("\nWord-by-word prediction:")
for i in range(len(tokens)-1):
    print(f"Input: {tokens[i]} → Predicted: {predicted_words[i]} (Expected: {tokens[i+1]})")

import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, SimpleRNN, Dense
from tensorflow.keras.utils import to_categorical

# 1. Take word input
word = input("Enter a word: ").lower()

# 2. Create letter-number maps
chars = sorted(set(word))
c2i = {c: i for i, c in enumerate(chars)}
i2c = {i: c for c, i in c2i.items()}

# 3. Prepare training data
X_seq = [c2i[c] for c in word[:-1]]
Y_seq = [c2i[c] for c in word[1:]]
X = to_categorical(X_seq, len(chars)).reshape(1, -1, len(chars))
Y = to_categorical(Y_seq, len(chars)).reshape(1, -1, len(chars))

# 4. Build model (using Functional API to access hidden states)
inputs = Input(shape=(None, len(chars)))
rnn, state_sequence = SimpleRNN(32, return_sequences=True, return_state=True)(inputs)
output = Dense(len(chars), activation='softmax')(rnn)
model = Model(inputs, output)

# 5. Compile and train
model.compile(loss='categorical_crossentropy', optimizer='adam')
model.fit(X, Y, epochs=300, verbose=0)

# 6. Create new model to extract hidden states
intermediate_model = Model(inputs=inputs, outputs=[rnn, output])
rnn_outputs, predictions = intermediate_model.predict(X)

# 7. Decode predictions
pred_indices = np.argmax(predictions, axis=2)[0]
predicted_letters = [i2c[i] for i in pred_indices]

# 8. Display results
print("\nOriginal word      :", word)
print("Predicted next     :", ''.join(predicted_letters))

print("\nLetter-by-letter prediction with hidden states:")
for t in range(len(X_seq)):
    input_char = word[t]
    expected_char = word[t+1]
    predicted_char = predicted_letters[t]
    hidden_state = rnn_outputs[0][t]  # hidden state at time t
    print(f"Input: {input_char} → Predicted: {predicted_char} (Expected: {expected_char})")
    print(f"  Hidden State[{t}]: {np.round(hidden_state[:8], 3)} ...\n")

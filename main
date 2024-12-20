import numpy as np
import tensorflow as tf
from music21 import converter, instrument, note, chord, stream
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

# Step 1: Parse MIDI files
def parse_midi(file):
    notes = []
    midi = converter.parse(file)
    for element in midi.flat.notes:
        if isinstance(element, note.Note):
            notes.append(str(element.pitch))
        elif isinstance(element, chord.Chord):
            notes.append('.'.join(str(n) for n in element.normalOrder))
    return notes

# Step 2: Create Sequences
def create_sequences(notes, sequence_length):
    note_to_int = {n: i for i, n in enumerate(sorted(set(notes)))}
    input_sequences = []
    output_notes = []
    for i in range(len(notes) - sequence_length):
        input_sequences.append([note_to_int[n] for n in notes[i:i + sequence_length]])
        output_notes.append(note_to_int[notes[i + sequence_length]])
    return np.array(input_sequences), np.array(output_notes)

# Step 3: Build the Model
def build_model(input_shape, output_dim):
    model = Sequential([
        LSTM(256, input_shape=input_shape, return_sequences=True),
        Dropout(0.3),
        LSTM(256),
        Dropout(0.3),
        Dense(256, activation='relu'),
        Dense(output_dim, activation='softmax')
    ])
    model.compile(loss='categorical_crossentropy', optimizer='adam')
    return model

# Step 4: Train the Model
def train_model(model, X, y, epochs=50, batch_size=64):
    model.fit(X, y, epochs=epochs, batch_size=batch_size, validation_split=0.2)

# Step 5: Generate Music
def generate_music(model, seed_sequence, note_to_int, int_to_note, length):
    output = []
    for _ in range(length):
        prediction = model.predict(seed_sequence)
        index = np.argmax(prediction)
        output.append(int_to_note[index])
        seed_sequence = np.roll(seed_sequence, -1, axis=1)
        seed_sequence[0, -1] = index
    return output

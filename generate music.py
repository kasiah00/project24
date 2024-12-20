import numpy as np
import tensorflow as tf
from music21 import converter, instrument, note, chord, stream
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Activation
from tensorflow.keras.utils import to_categorical

def generate_music(model, seed_sequence, note_to_int, int_to_note, length):
    """Generates music using the trained model and a seed sequence."""
    output = []
    sequence = seed_sequence
    for _ in range(length):
        prediction = model.predict(sequence, verbose=0)
        index = np.argmax(prediction)
        result = int_to_note[index]
        output.append(result)

        sequence = np.roll(sequence, -1, axis=1)
        sequence[0, -1, 0] = index / float(len(note_to_int))
    return output

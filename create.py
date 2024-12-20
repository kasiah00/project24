import numpy as np
import tensorflow as tf
from music21 import converter, instrument, note, chord, stream
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Activation
from tensorflow.keras.utils import to_categorical

def create_sequences(notes, sequence_length):
    """Creates input and output sequences for the model."""
    note_to_int = {n: i for i, n in enumerate(sorted(set(notes)))}
    int_to_note = {i: n for n, i in note_to_int.items()}

    input_sequences = []
    output_notes = []

    for i in range(len(notes) - sequence_length):
        input_sequences.append([note_to_int[n] for n in notes[i:i + sequence_length]])
        output_notes.append(note_to_int[notes[i + sequence_length]])

    input_sequences = np.reshape(input_sequences, (len(input_sequences), sequence_length, 1))
    input_sequences = input_sequences / float(len(note_to_int))
    output_notes = to_categorical(output_notes, num_classes=len(note_to_int))

    return input_sequences, output_notes, note_to_int, int_to_note

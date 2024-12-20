import numpy as np
import tensorflow as tf
from music21 import converter, instrument, note, chord, stream
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Activation
from tensorflow.keras.utils import to_categorical

def train_model(model, X, y, epochs=50, batch_size=64):
    """Trains the model using the input and output sequences."""
    model.fit(X, y, epochs=epochs, batch_size=batch_size, validation_split=0.2)

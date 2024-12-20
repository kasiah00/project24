def create_sequences(notes, sequence_length):
    note_to_int = {n: i for i, n in enumerate(sorted(set(notes)))}
    input_sequences = []
    output_notes = []
    for i in range(len(notes) - sequence_length):
        input_sequences.append([note_to_int[n] for n in notes[i:i + sequence_length]])
        output_notes.append(note_to_int[notes[i + sequence_length]])
    return np.array(input_sequences), np.array(output_notes)

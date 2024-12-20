if __name__ == "__main__":
    # Parameters
    midi_files = ["path/to/midi1.mid", "path/to/midi2.mid"]
    sequence_length = 50
    generated_length = 500

    # Step 1: Parse MIDI files
    all_notes = []
    for file in midi_files:
        all_notes.extend(parse_midi(file))

    # Step 2: Create sequences
    X, y, note_to_int, int_to_note = create_sequences(all_notes, sequence_length)

    # Step 3: Build model
    model = build_model(X.shape[1:], len(note_to_int))

    # Step 4: Train model
    train_model(model, X, y, epochs=100, batch_size=64)

    # Step 5: Generate music
    seed = X[np.random.randint(0, len(X) - 1)].reshape(1, sequence_length, 1)
    prediction = generate_music(model, seed, note_to_int, int_to_note, generated_length)

    # Step 6: Save to MIDI
    create_midi(prediction, "output_music.mid")

def generate_music(model, seed_sequence, note_to_int, int_to_note, length):
    output = []
    for _ in range(length):
        prediction = model.predict(seed_sequence)
        index = np.argmax(prediction)
        output.append(int_to_note[index])
        seed_sequence = np.roll(seed_sequence, -1, axis=1)
        seed_sequence[0, -1] = index
    return output

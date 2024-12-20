def parse_midi(file):
    notes = []
    midi = converter.parse(file)
    for element in midi.flat.notes:
        if isinstance(element, note.Note):
            notes.append(str(element.pitch))
        elif isinstance(element, chord.Chord):
            notes.append('.'.join(str(n) for n in element.normalOrder))
    return notes

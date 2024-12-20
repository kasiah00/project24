def train_model(model, X, y, epochs=50, batch_size=64):
    model.fit(X, y, epochs=epochs, batch_size=batch_size, validation_split=0.2)

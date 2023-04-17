import numpy as np
import keras
from keras.layers import LSTM, Dense
from keras.models import Sequential

# Define the model
model = Sequential()
model.add(LSTM(128, input_shape=(100, 1), return_sequences=True))
model.add(LSTM(128))
model.add(Dense(1, activation='tanh'))
model.compile(loss='mean_absolute_error', optimizer='adam')

# Load the training data
data = np.load('music_data.npy')
x_train = data[:-1]
y_train = data[1:]

# Reshape the data for LSTM
x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

# Train the model
model.fit(x_train, y_train, epochs=50, batch_size=32)

# Generate new music
seed = np.random.rand(100, 1)
generated_music = []
for i in range(100):
    prediction = model.predict(np.array([seed]))
    generated_music.append(prediction)
    seed = np.vstack((seed[1:], prediction))

# Save the generated music as a MIDI file
from midiutil.MidiFile import MIDIFile

midi_file = MIDIFile(1)
track = 0
time = 0
midi_file.addTrackName(track, time, "Generated Music")
midi_file.addTempo(track, time, 120)

channel = 0
volume = 100
for i, note in enumerate(generated_music):
    midi_file.addNote(track, channel, int(note * 127), time, i * 0.5, 0.5)

with open("generated_music.mid", "wb") as output_file:
    midi_file.writeFile(output_file)

import collections
#import datetime
#import fluidsynth
import glob
import numpy as np
import os
import pandas as pd
import pretty_midi
import tensorflow as tf
from tensorflow.keras import layers, Sequential



class RNN_Model(object):

    def __init__(self, learning_rate=0.005, seq_length=25 ):

        ##
        seed = 42
        tf.random.set_seed(seed)
        np.random.seed(seed)

        # Sampling rate for audio playback
        self._SAMPLING_RATE = 16000
        self.learning_rate = learning_rate
        self.key_order = ['pitch', 'step', 'duration']
        self.seq_length = seq_length

        self.raw_notes = []

        inputs = tf.keras.Input((self.seq_length, 3))
        x = tf.keras.layers.LSTM(128)(inputs)

        outputs = {
        'pitch': tf.keras.layers.Dense(128, name='pitch')(x),
        'step': tf.keras.layers.Dense(1, name='step')(x),
        'duration': tf.keras.layers.Dense(1, name='duration')(x),
        }

        self.model = tf.keras.Model(inputs, outputs)


    def compile_model(self):


        def mse_with_positive_pressure(y_true: tf.Tensor, y_pred: tf.Tensor):
            mse = (y_true - y_pred) ** 2
            positive_pressure = 10 * tf.maximum(-y_pred, 0.0)
            return tf.reduce_mean(mse + positive_pressure)

        loss = {
            'pitch': tf.keras.losses.SparseCategoricalCrossentropy(
                from_logits=True),
            'step': mse_with_positive_pressure,
            'duration': mse_with_positive_pressure,
        }

        optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate)

        self.model.compile(
            loss=loss,
            loss_weights={
                'pitch': 0.05,
                'step': 1.0,
                'duration':1.0,
            },
            optimizer=optimizer,
        )

    def fit_data(self, train_ds, epochs = 50):

        callbacks = [
            tf.keras.callbacks.ModelCheckpoint(
                filepath='./training_checkpoints/ckpt_{epoch}',
                save_weights_only=True),
            tf.keras.callbacks.EarlyStopping(
                monitor='loss',
                patience=5,
                verbose=1,
                restore_best_weights=True),
        ]

        history = self.model.fit(
            train_ds,
            epochs=epochs,
            callbacks=callbacks,
        )

        return history

    def evaluate(self, train_ds):

        return self.evaluate(train_ds, return_dict=True)

    def process_data(self, filenames,
                    num_files = 5,
                    vocab_size = 128,
                    ):

        all_notes = []

        for f in filenames[:num_files]:
            notes = self._midi_to_notes(f)
            all_notes.append(notes)

        all_notes = pd.concat(all_notes)

        n_notes = len(all_notes)

        train_notes = np.stack([all_notes[key] for key in self.key_order], axis=1)
        notes_ds = tf.data.Dataset.from_tensor_slices(train_notes)
        notes_ds.element_spec

        batch_size = 64
        buffer_size = n_notes - self.seq_length  # the number of items in the dataset\
        seq_ds = self._create_sequences(notes_ds, self.seq_length, vocab_size)
        seq_ds.element_spec

        train_ds = (seq_ds
                    .shuffle(buffer_size)
                    .batch(batch_size, drop_remainder=True)
                    .cache()
                    .prefetch(tf.data.experimental.AUTOTUNE))

        print(' --- Training Data Ready --- ')
        return train_ds


    def _midi_to_notes(self, midi_file: str) -> pd.DataFrame:
        pm = pretty_midi.PrettyMIDI(midi_file)
        instrument = pm.instruments[0]
        notes = collections.defaultdict(list)

        # Sort the notes by start time
        sorted_notes = sorted(instrument.notes, key=lambda note: note.start)
        prev_start = sorted_notes[0].start

        for note in sorted_notes:
            start = note.start
            end = note.end
            notes['pitch'].append(note.pitch)
            notes['start'].append(start)
            notes['end'].append(end)
            notes['step'].append(start - prev_start)
            notes['duration'].append(end - start)
            prev_start = start

        return pd.DataFrame({name: np.array(value) for name, value in notes.items()})

    def _notes_to_midi(self,
        notes: pd.DataFrame,
        out_file: str,
        instrument_name: str,
        velocity: int = 100,  # note loudness
        ) -> pretty_midi.PrettyMIDI:

        pm = pretty_midi.PrettyMIDI()
        instrument = pretty_midi.Instrument(
            program=pretty_midi.instrument_name_to_program(
                instrument_name))

        prev_start = 0
        for i, note in notes.iterrows():
            start = float(prev_start + note['step'])
            end = float(start + note['duration'])
            note = pretty_midi.Note(
                velocity=velocity,
                pitch=int(note['pitch']),
                start=start,
                end=end,
            )
            instrument.notes.append(note)
            prev_start = start

        pm.instruments.append(instrument)
        pm.write(out_file)
        return pm

    def _create_sequences(self,
        dataset: tf.data.Dataset,
        seq_length: int,
        vocab_size = 128,
        ) -> tf.data.Dataset:

        seq_length = seq_length+1

        # Take 1 extra for the labels
        windows = dataset.window(seq_length, shift=1, stride=1,
                                    drop_remainder=True)

        # `flat_map` flattens the" dataset of datasets" into a dataset of tensors
        flatten = lambda x: x.batch(seq_length, drop_remainder=True)
        sequences = windows.flat_map(flatten)

        # Normalize note pitch
        def scale_pitch(x):
            x = x/[vocab_size,1.0,1.0]
            return x

        # Split the labels
        def split_labels(sequences):
            inputs = sequences[:-1]
            labels_dense = sequences[-1]
            labels = {key:labels_dense[i] for i,key in enumerate(self.key_order)}

            return scale_pitch(inputs), labels

        return sequences.map(split_labels, num_parallel_calls=tf.data.AUTOTUNE)

    def _predict_next_note(
        self,
        notes: np.ndarray,
        temperature: float = 1.0) -> tuple[int, float, float]:

        assert temperature > 0

        # Add batch dimension
        inputs = tf.expand_dims(notes, 0)

        predictions = self.model.predict(inputs)
        pitch_logits = predictions['pitch']
        step = predictions['step']
        duration = predictions['duration']

        pitch_logits /= temperature
        pitch = tf.random.categorical(pitch_logits, num_samples=1)
        pitch = tf.squeeze(pitch, axis=-1)
        duration = tf.squeeze(duration, axis=-1)
        step = tf.squeeze(step, axis=-1)

        # `step` and `duration` values should be non-negative
        step = tf.maximum(0, step)
        duration = tf.maximum(0, duration)

        return int(pitch), float(step), float(duration)

    def generate_notes_from_midi_file(self, sample_file, seq_length=25 ,vocab_size = 128, temperature = 2.0, num_predictions = 120):

        raw_notes = self._midi_to_notes(sample_file)
        sample_notes = np.stack([raw_notes[key] for key in self.key_order], axis=1)

        # The initial sequence of notes; pitch is normalized similar to training
        # sequences
        input_notes = (
            sample_notes[:seq_length] / np.array([vocab_size, 1, 1]))

        generated_notes = []
        prev_start = 0
        for _ in range(num_predictions):
            pitch, step, duration = self._predict_next_note(input_notes, temperature)
            start = prev_start + step
            end = start + duration
            input_note = (pitch, step, duration)
            generated_notes.append((*input_note, start, end))
            input_notes = np.delete(input_notes, 0, axis=0)
            input_notes = np.append(input_notes, np.expand_dims(input_note, 0), axis=0)
            prev_start = start

        generated_notes = pd.DataFrame(
            generated_notes, columns=(*self.key_order, 'start', 'end'))

        return generated_notes





if __name__ == "__main__":


    #find midi data in directory
    my_data = []

    path = './maestro-v3.0.0-midi/maestro-v3.0.0/2018/'
    for file in os.listdir(path):
        if file.endswith('.midi'):
            my_data.append(path+file)

    #instantiate model
    RNN = RNN_Model()

    RNN.compile_model()

    #process and train data
    train_ds = RNN.process_data(my_data)

    history = RNN.fit_data(train_ds, epochs=1) #epochs = 50 is best

    #sample for predicting next notes
    sample = my_data[1]

    #get instrument data from sample
    pm = pretty_midi.PrettyMIDI(sample)
    instrument = pm.instruments[0]
    instrument_name = pretty_midi.program_to_instrument_name(instrument.program)

    #generate notes from model
    gen_notes = RNN.generate_notes_from_midi_file(my_data[1])

    # now call 'notes to midi' on this file
    print(gen_notes)

    #or write file
    out_file = 'output.mid'
    out_pm = RNN._notes_to_midi(
        gen_notes, out_file=out_file, instrument_name=instrument_name)

    print(" --- completed ---")

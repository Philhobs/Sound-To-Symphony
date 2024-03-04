import tensorflow as tf
from tensorflow.keras import layers, losses, Model
import librosa
import numpy as np
import soundfile as sf
import time
import os

class Autoencoder(Model):
    def __init__(self, input_shape=(700, 33), latent_dim=300):
        super(Autoencoder, self).__init__()
        self.latent_dim = latent_dim
        self.sample_rate = 22050
        self.model_input_shape = input_shape

        # Encoder architecture
        self.encoder = tf.keras.Sequential([
            layers.Conv1D(2, 3, activation='relu', input_shape=input_shape),
            layers.Flatten(),
            layers.Dense(latent_dim, activation='relu'),
        ])

        # Decoder architecture
        self.decoder = tf.keras.Sequential([
            layers.Dense(tf.math.reduce_prod(input_shape), activation='linear'),
            layers.Reshape(input_shape),
        ])

    def call(self, inputs):
        encoded = self.encoder(inputs)
        decoded = self.decoder(encoded)
        return decoded

    def compile_model(self, optimizer='adam', loss=losses.MeanAbsoluteError()):
        self.compile(optimizer=optimizer, loss=loss)

    def fit_training_sound(self, filename, epochs=100, batch_size=32, samples=100, snippet_len=128*128, mels=700):
        sound, sr = librosa.load(filename)
        sound_matrix_size = int((len(sound) / snippet_len) // samples)

        if sound_matrix_size < 1:
            print("Not enough information. Please input a larger sound file.")
            return

        sound_trimmed = sound[:sound_matrix_size * samples * snippet_len]
        X_train = sound_trimmed.reshape(sound_matrix_size * samples, snippet_len)

        # Generate Mel spectrograms
        X_train_mels = librosa.feature.melspectrogram(y=X_train, sr=sr, n_fft=2500, hop_length=500, n_mels=mels)
        X_train_mels_db = librosa.power_to_db(X_train_mels)

        print("----Fitting Data----")
        self.fit(X_train_mels_db, X_train_mels_db, epochs=epochs, batch_size=batch_size, shuffle=True, verbose=1)

    def warp_sample(self, filename, snippet_len=128*128):
        sample, sr = librosa.load(filename)
        sample_size = int(len(sample) / snippet_len // 1)

        if sample_size < 1:
            print("Audio sample too small.")
            return

        print("~~~~Warping sample~~~~")
        sample_trimmed = sample[:snippet_len]
        sample_mels = librosa.feature.melspectrogram(y=sample_trimmed, sr=sr, n_fft=2500, hop_length=500, n_mels=700)
        sample_mels_db = librosa.power_to_db(sample_mels)
        sample_mels_db_reshaped = sample_mels_db.reshape(1, *self.model_input_shape)

        encoded_mel = self.encoder(sample_mels_db_reshaped)
        decoded_mel = self.decoder(encoded_mel).numpy().squeeze(0)

        warped_sample = librosa.feature.inverse.mel_to_audio(decoded_mel, n_fft=2500, sr=sr)
        return warped_sample

    def save_model(self, file_path='autoencoder_model'):
        """
        Saves the entire model to a file.
        """
        self.save(file_path, save_format='tf')

    @staticmethod
    def load_model(file_path='autoencoder_model.tf'):
        """
        Loads a model from the specified file path.
        """
        return tf.keras.models.load_model(file_path, custom_objects={'Autoencoder': Autoencoder})

if __name__ == "__main__":
    print(f"Current Working Directory: {os.getcwd()}")
    autoencoder = Autoencoder()
    autoencoder.compile_model()

    base_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(base_dir, 'audio_bank', 'AMINOR.wav')

    # Make sure you're using 'file_path' here and not the incorrect relative path
    autoencoder.fit_training_sound(file_path, epochs=10)
    autoencoder.save_model('src/model/weights/autoencoder_model.tf')


    #this needs to be in the streamlit
    # warped = autoencoder.warp_sample('./webapp/upload/SYNTH_piano.mp3')
    # sf.write(f"./webapp/autoencoded_sound/warped_sound_{time.time()}.wav", warped, autoencoder.sample_rate)

import tensorflow as tf
from tensorflow.keras import layers, losses, Model
import librosa
import numpy as np
import soundfile as sf
import time

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

    def fit_training_sound(self, filename, epochs=100, batch_size=64, samples=100, snippet_len=128*128, mels=700):
        sound, sr = librosa.load(filename)
        sound_matrix_size = int((len(sound) / snippet_len) // samples)

        if sound_matrix_size < 1:
            print("Not enough information. Please input a larger sound file.")
            return

        sound_trimmed = sound[:sound_matrix_size * samples * snippet_len]
        X_train = sound_trimmed.reshape(sound_matrix_size * samples, snippet_len)

        # Generate Mel spectrograms
        X_train_mels = librosa.feature.melspectrogram(y=X_train, sr=sr, n_fft=2400, hop_length=500, n_mels=mels)
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
        sample_mels = librosa.feature.melspectrogram(y=sample_trimmed, sr=sr, n_fft=2400, hop_length=500, n_mels=700)
        sample_mels_db = librosa.power_to_db(sample_mels)
        sample_mels_db_reshaped = sample_mels_db.reshape(1, *self.model_input_shape)

        encoded_mel = self.encoder(sample_mels_db_reshaped)
        decoded_mel = self.decoder(encoded_mel).numpy().squeeze(0)

        warped_sample = librosa.feature.inverse.mel_to_audio(decoded_mel, n_fft=2400, sr=sr)
        return warped_sample

    def generate_random_sounds(self, n_samples, mean=0, scale=1):
        sample_list = []
        print("~~~~Generating noise samples~~~~")
        for n in range(n_samples):
            normal_noise = np.random.normal(loc=mean, scale=scale, size=(1, self.latent_dim))
            decoded_mel = self.decoder(normal_noise).numpy().squeeze(0)
            random_sound = librosa.feature.inverse.mel_to_audio(decoded_mel,  n_fft=2400, sr=self.sample_rate)
            sample_list.append(random_sound)

        return sample_list

if __name__ == "__main__":
    autoencoder = Autoencoder(latent_dim=300)
    autoencoder.compile_model()

    ''' Train the model with a sound '''
    autoencoder.fit_training_sound('./audio_bank/SYNTH_track.wav', epochs=250, samples=10)

    ''' Create hybrid sound '''
    #warped = autoencoder.warp_sample('./audio_bank/SYNTH_sample.wav')
    #sf.write(f"./audio_bank/warped_sound_{time.time()}.wav", warped, autoencoder.sample_rate)

    ''' Generate random sounds '''
    #warped_noise_sample = autoencoder.generate_random_sounds(n_samples=2, mean=6, scale=1)

    #for x in range(len(warped_noise_sample)):
        #sf.write(f"./audio_bank/warped_sound_{time.time()}.wav", warped_noise_sample[x], autoencoder.sample_rate)

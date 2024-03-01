import tensorflow as tf
import librosa
import numpy as np
from tensorflow.keras import layers, losses
from tensorflow.keras.models import Model

class Autoencoder(Model):
    def __init__(self, shape=(700, 33), latent_dim=300):
        super(Autoencoder, self).__init__()

        self.latent_dim = latent_dim
        self.SAMPLE_RATE = 22050
        self.shape = shape
        self.encoder = tf.keras.Sequential([

          layers.Conv1D(2, 3, activation='relu', input_shape=shape),

          # For your consideration layers to implement.
          #layers.Conv1D(3, 6, activation='relu'),
          #layers.Dense(1000, activation='linear'),
          #layers.Dense(25, activation='relu'), #relu?

          layers.Flatten(),
          layers.Dense(latent_dim, activation='relu'), #relu?
        ])

        self.decoder = tf.keras.Sequential([

          #layers.Conv1DTranspose(16, 6, activation='relu'),
          #layers.Conv1DTranspose(32, 3, activation='relu'),

          layers.Dense(tf.math.reduce_prod(shape), trainable=True, activation = 'linear'),
          layers.Reshape(shape),
        ])

    def call(self, x):
        '''also used when .fit() ?'''
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

    def compile_me(self, optimizer='adam', loss=losses.MeanAbsoluteError()):

        self.compile(optimizer=optimizer, loss=loss)


    def fit_training_sound(self, filename,
                            verbose = 1,
                            epochs = 100,
                            batch_size = 32,
                            samples = 100,
                            snippet_len = 128*128,
                            mels = 700):
        ''' mels and snippets define the SHAPE information
            going to the autoencoder. This SHAPE information
            can be dynamic.

            More mels implies higher resolution sound.
            hop length with affect the time.

        '''
        sound, sr = librosa.load(filename)

        sound_matrix_size = int(((sound.shape[0] / snippet_len) // samples))

        if sound_matrix_size < 1: #is not_large_enough
            print("Not enough information. Input larger sound file.")

        else:

            #edit sound,
            range = sound_matrix_size*samples*snippet_len

            X_train = sound[:range].reshape(sound_matrix_size*samples, snippet_len)

            #generate spectograms of the sound snippets
            X_train_mels = librosa.feature.melspectrogram(
                    y=X_train,
                    sr=sr, n_fft=2500,
                    hop_length=500,
                    n_mels=mels)

            #decibel scaler, essentially log scaling.
            X_train_mels_db = librosa.power_to_db(X_train_mels)

            try:

                print("""
                      ----Fitting Data----
                      """)
                self.fit(X_train_mels_db,
                         X_train_mels_db,
                        epochs=epochs,
                        batch_size = batch_size,
                        shuffle=True,
                        verbose = verbose)

            except:
                print("""something went wrong while fitting.
                         check that the shape is correct.""")


    def warp_sample(self, filename, snippet_len = 128*128,):

        sample, sr = librosa.load(filename)

        sample_size = int(sample.shape[0] / (128*128) // 1 )

        if sample_size < 1:
            print("Audio sample too small.")

        else:
            print("~~~~warping sample~~~~")
            sample_edit = sample[:snippet_len]
            sample_mels = librosa.feature.melspectrogram(y=sample_edit,
                                            sr=sr, n_fft=2500,
                                            hop_length=500,
                                            n_mels=700)

            sample_mels_db = librosa.power_to_db(sample_mels)

            sample_mels_db_reshaped = sample_mels_db.reshape(1, 700, 33)

            encoded_mel = self.encoder(sample_mels_db_reshaped)
            decoded_mel = self.decoder(encoded_mel)

            decoded_mel = np.squeeze(decoded_mel, axis=0)

            warped_sample = librosa.feature.inverse.mel_to_audio(decoded_mel,  n_fft=2500, sr=sr)

            return warped_sample



if __name__ == "__main__":

    import time
    import soundfile as sf
    auto_enc = Autoencoder()
    auto_enc.compile_me()

    auto_enc.fit_training_sound('./AMINOR.wav',epochs=10)
    warped = auto_enc.warp_sample('./SYNTH_piano.mp3')

    sf.write(f"warped_sound_{time.time()}.wav", warped, auto_enc.SAMPLE_RATE)

    #ipd.Audio(warped, rate=)
    #print(auto_enc.encoder.summary())
    #print(auto_enc.decoder.summary())

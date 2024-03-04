import streamlit as st
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import os
from IPython.display import Audio
import soundfile as sf
from autoencoder_model import Autoencoder
import tensorflow as tf

def generate_colored_text(text, colors):
    return "".join(
        f"<span style='color: {colors[i % len(colors)]}; font-size: 75px; font-weight: bold;'>{char}</span>"
        for i, char in enumerate(text)
    )

def align_text(text, alignment='center'):
    return f"<div style='text-align: {alignment};'>{text}</div>"

# rainbow_colors = ['#693C72', '#C15050', '#D97642', '#337357', '#D49D42']
rainbow_colors = ['#54478c', '#2c699a', '#048ba8', '#0db39e', '#16db93', '#83e377']
title = "Sound to Sympho♫y"
notes = "♩♫♪"

# Centered title
colored_title = generate_colored_text(title, rainbow_colors)
st.markdown(align_text(colored_title, 'center'), unsafe_allow_html=True)

#todo -- make name dynamic in case of multiple files
def plot_spectrogram(audio_path, output_path='upload/spectrogram.png', cmap='viridis'):
    y, sr = librosa.load(audio_path)
    S = librosa.feature.melspectrogram(y=y, sr=sr)
    S_dB = librosa.power_to_db(S, ref=np.max)
    plt.figure(figsize=(10, 4))
    librosa.display.specshow(S_dB, sr=sr, x_axis='time', y_axis='mel', cmap=cmap)
    plt.colorbar(format='%+2.0f dB')
    plt.title('Mel-frequency spectrogram')
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

uploaded_file = st.file_uploader(" ",type=["wav", "mp3"])

#space
st.text("")
st.write("<h1 style='font-size: 36px;'>Spectrogram</h1>", unsafe_allow_html=True)

if uploaded_file is not None:
    file_name = uploaded_file.name
    file_path = os.path.join('upload', file_name)
    with open(file_path, 'wb') as f:
        f.write(uploaded_file.read())
        # creates spectrogram
    plot_spectrogram(file_path)
    st.image('upload/spectrogram.png', caption='Mel-frequency spectrogram')

    autoencoder = Autoencoder()
    autoencoder.load_model('/Users/christellejoseph/S2S/Sound-To-Symphony/src/model/weights/autoencoder_model.tf')

    warped = autoencoder.warp_sample(file_path)
    sf.write(f"{file_path}.wav", warped, autoencoder.sample_rate)
    new_sound = f"{file_path}.wav"
    print('new_songs', new_sound)

    cols = st.columns([1])
    col1 = cols[0]
    col1.write("#### Piano")
    col1.audio(new_sound)

else:
    print("No file was uploaded.")

# col1 = st.columns([1])
# if new_song is not None:
#     with col1:
#         st.write("#### Piano")
#         st.audio('upload/example.wav')

# col1, col2, col3  = st.columns([0.8, 0.8, 0.8])
# with col1:
#     st.write("#### Piano")
#     st.audio('upload/example.wav')

# with col2:
#     st.write("#### Guitar")
#     st.audio('upload/example.wav')
# with col3:
#     st.write("#### Synth")
#     st.audio('upload/example.wav')

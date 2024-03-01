import streamlit as st
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import os
from IPython.display import Audio

rainbow_colors = ['#693C72', '#C15050', '#D97642', '#337357', '#D49D42']
title = "Sound to Symphony"
colored_title = "".join(
    f"<span style='color: {rainbow_colors[i % len(rainbow_colors)]}; font-size: 60px; font-weight: bold;'>{char}</span>"
    for i, char in enumerate(title)
)
st.markdown(colored_title, unsafe_allow_html=True)

def plot_spectrogram(audio_path):
    y, sr = librosa.load(audio_path)
    S = librosa.feature.melspectrogram(y=y, sr=sr)
    S_dB = librosa.power_to_db(S, ref=np.max)
    plt.figure(figsize=(10, 4))
    librosa.display.specshow(S_dB, sr=sr, x_axis='time', y_axis='mel')
    plt.colorbar(format='%+2.0f dB')
    plt.title('Mel-frequency spectrogram')
    plt.tight_layout()
    plt.savefig('upload/spectrogram.png')
    plt.close()


uploaded_file = st.file_uploader(" ",type=["wav"])

#space
st.text("")
st.write("<h1 style='font-size: 36px;'>Spectrogram</h1>", unsafe_allow_html=True)

if uploaded_file is not None:
    file_name = uploaded_file.name
    file_path = os.path.join('upload', file_name)
    with open(file_path, 'wb') as f:
        f.write(uploaded_file.read())
        # Generate spectrogram
    plot_spectrogram(file_path)
    st.image('upload/spectrogram.png', caption='Mel-frequency spectrogram')
else:
    print("No file was uploaded.")


col1, col2, col3  = st.columns([0.8, 0.8, 0.8])
with col1:
    st.write("#### Piano")
    st.audio('upload/example.wav')
with col2:
    st.write("#### Guitar")
    st.audio('upload/example.wav')
with col3:
    st.write("#### Synth")
    st.audio('upload/example.wav')

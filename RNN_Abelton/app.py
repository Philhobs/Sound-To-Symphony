import streamlit as st
from pitch_list import pitch
from RNN_model import RNN_Model
import tensorflow as tf
import pretty_midi
import os
import io

def generate_colored_text(text, colors):
    return "".join(
        f"<span style='color: {colors[i % len(colors)]}; font-size: 75px; font-weight: bold;'>{char}</span>"
        for i, char in enumerate(text)
    )

def align_text(text, alignment='center'):
    return f"<div style='text-align: {alignment};'>{text}</div>"

rainbow_colors = ['#693C72', '#C15050', '#D97642', '#337357', '#D49D42']
# rainbow_colors = ['#54478c', '#2c699a', '#048ba8', '#0db39e', '#16db93', '#83e377']
title = "Sound to Sympho♫y"
notes = "♩♫♪"
colored_title = generate_colored_text(title, rainbow_colors)
st.markdown(align_text(colored_title, 'center'), unsafe_allow_html=True)

#***** code *****

## construct model
rnn = RNN_Model()
rnn.compile_model()
checkpoint_path = './training_checkpoints/ckpt_5'
rnn.model.load_weights(checkpoint_path)


uploaded_file = st.file_uploader(" ",type=["mid"])

if uploaded_file is not None:
    input_file = uploaded_file.name
    file_path = os.path.join('', input_file)
    with open(file_path, 'wb') as f:
        f.write(uploaded_file.read())
    pm = pretty_midi.PrettyMIDI(file_path)
    instrument = pm.instruments[0]
    instrument_name = pretty_midi.program_to_instrument_name(instrument.program)

option = st.selectbox(
   "Select a pitch",
   pitch,
   index=None,
#    placeholder="Select pitch",
)
st.write('You selected:', option)

st.write("")


key_name = ''
if st.button('Generate Midi'):
    gen_notes = rnn.generate_notes_from_midi_file(file_path, key=option)
    output_path = 'output.mid'
    output_midi = rnn._notes_to_midi(gen_notes, out_file=output_path, instrument_name=instrument_name)


    # Create an in-memory bytes buffer for your MIDI file
    with open(output_path, 'rb') as f:
        midi_bytes = f.read()
    midi_buffer = io.BytesIO(midi_bytes)

    st.audio(data=output_path, format='audio/midi', start_time=0)

    st.download_button(
         label="Download Midi",
         data=midi_buffer,
         file_name="generated_midi.mid",
         mime="audio/midi"
    )

import streamlit as st
from pitch_list import pitch
from RNN_model import RNN_Model
import tensorflow as tf



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



option = st.selectbox(
   "Select a pitch",
   pitch,
   index=None,
#    placeholder="Select pitch",
)
st.write('You selected:', option)

st.write("")
st.button('Generate Music')

model = RNN_Model()
model.compile_model()

checkpoint_path = './training_checkpoints/ckpt_5'
model.load_weights(checkpoint_path)

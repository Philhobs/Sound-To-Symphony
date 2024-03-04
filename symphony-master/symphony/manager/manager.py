import pretty_midi
from IPython.display import Audio
import pandas as pd
from typing import Optional
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import collections

class AudioManager:
    @staticmethod
    def display_audio(pm: pretty_midi.PrettyMIDI, sampling_rate=44100, seconds=30):
        """
        Display audio generated from a PrettyMIDI object.

        Parameters:
        - pm: pretty_midi.PrettyMIDI
            PrettyMIDI object containing MIDI data.
        - sampling_rate: int
            Sampling rate for the audio (default is 44100).
        - seconds: int
            Number of seconds to display audio for (default is 30).

        Returns:
        - IPython.display.Audio
            Audio object to be displayed in IPython environment.
        """
        waveform = pm.fluidsynth(fs=sampling_rate)
        # Take a sample of the generated waveform to mitigate kernel resets
        waveform_short = waveform[:seconds * sampling_rate]
        return Audio(waveform_short, rate=sampling_rate)

    @staticmethod
    def plot_piano_roll(notes: pd.DataFrame, count: Optional[int] = None):
        if count:
            title = f'First {count} notes'
        else:
            title = f'Whole track'
            count = len(notes['pitch'])
        plt.figure(figsize=(20, 4))
        plot_pitch = np.stack([notes['pitch'], notes['pitch']], axis=0)
        plot_start_stop = np.stack([notes['start'], notes['end']], axis=0)
        plt.plot(
            plot_start_stop[:, :count], plot_pitch[:, :count], color="b", marker=".")
        plt.xlabel('Time [s]')
        plt.ylabel('Pitch')
        _ = plt.title(title)

    @staticmethod
    def plot_distributions(notes: pd.DataFrame, drop_percentile=2.5):
        plt.figure(figsize=[15, 5])
        plt.subplot(1, 3, 1)
        sns.histplot(notes, x="pitch", bins=20)

        plt.subplot(1, 3, 2)
        max_step = np.percentile(notes['step'], 100 - drop_percentile)
        sns.histplot(notes, x="step", bins=np.linspace(0, max_step, 21))

        plt.subplot(1, 3, 3)
        max_duration = np.percentile(notes['duration'], 100 - drop_percentile)
        sns.histplot(notes, x="duration", bins=np.linspace(0, max_duration, 21))



if __name__ == "__main__":
    # You can add test code here if needed
    pass

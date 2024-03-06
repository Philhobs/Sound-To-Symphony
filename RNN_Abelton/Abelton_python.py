import os
import time
import mido
import rtmidi
from threading import Thread
from mido import MidiFile
from RNN_Original import RNN_Model

# Assuming RNN_Model class is defined elsewhere and imported here

class ContinuousMidiPlayer:
    def __init__(self, model, base_output_path):
        self.model = model
        self.base_output_path = base_output_path
        self.midiout = rtmidi.MidiOut()
        self.setup_midi_output()

    def setup_midi_output(self):
        available_ports = self.midiout.get_ports()
        if available_ports:
            self.midiout.open_port(0)
        else:
            self.midiout.open_virtual_port("Virtual MIDI Output")

    def generate_and_save_midi(self, sample_path):
        timestamp = int(time.time())
        output_path = os.path.join(self.base_output_path, f"generated_{timestamp}.mid")
        gen_notes = self.model.generate_notes_from_midi_file(sample_path)
        self.model._notes_to_midi(gen_notes, out_file=output_path, instrument_name="Acoustic Grand Piano")
        return output_path

    def play_midi_file(self, midi_path):
        mid = MidiFile(midi_path)
        for msg in mid.play():
            if not msg.is_meta:
                midi_bytes = msg.bytes()
                self.midiout.send_message(midi_bytes)

    def ensure_midi_range(value):

        return max(0, min(127, value))



    def continuous_playback(self, initial_midi_path):
        current_midi_path = initial_midi_path

        while True:
            # Start playing the current MIDI file in a new thread
            Thread(target=self.play_midi_file, args=(current_midi_path,)).start()

            # Generate the next MIDI file while the current one is playing
            next_midi_path = self.generate_and_save_midi(current_midi_path)

            # Wait for the current MIDI file to finish playing
            mid_length = mido.MidiFile(current_midi_path).length
            time.sleep(mid_length)

            # Update the path for the next iteration
            current_midi_path = next_midi_path

if __name__ == "__main__":
    model = RNN_Model()  # Assume your RNN_Model is initialized and ready
    player = ContinuousMidiPlayer(model, "/Users/path/to/the/midi_output/")
    initial_midi_path = "/Users/path/to/the/output.mid"  # Set this to the first MIDI file (maybe we can choose one, already uploaded on the streamlite?)
    player.continuous_playback(initial_midi_path)

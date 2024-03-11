import matplotlib.pyplot as plt
import numpy as np

def plot_signal_with_speaker_diarization(signal, segments, sample_rate):
    plt.figure(figsize=(10, 4))

    # Plot the signal
    time = np.arange(len(signal)) / sample_rate
    plt.plot(time, signal, color='black')

    # Overlay speaker diarization segments
    for segment in segments:
        start_time, end_time, speaker_id = segment
        segment_signal = signal[int(start_time * sample_rate):int(end_time * sample_rate)]
        segment_time = np.linspace(start_time, end_time, len(segment_signal))
        plt.plot(segment_time, segment_signal, color='C'+str(speaker_id), linewidth=2)
        # Add label for each segment
        label_x = start_time
        label_y = np.max(signal) * 0.8
        plt.text(label_x, label_y, f'Speaker {speaker_id}', ha='left', va='center', color='C'+str(speaker_id))

    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.title('Speaker Diarization on Audio Signal')
    plt.grid(True)
    plt.show()

# Example audio signal (replace with your own)
sample_rate = 44100
duration = 10  # seconds
time = np.linspace(0, duration, sample_rate * duration)
signal = np.sin(2 * np.pi * 440 * time) + np.sin(2 * np.pi * 880 * time)  # Example signal with two frequencies

# Example speaker diarization segments (replace with your own)
segments = [(0, 2, 0), (2, 5, 1), (5, 8, 0), (8, 10, 2)]

plot_signal_with_speaker_diarization(signal, segments, sample_rate)

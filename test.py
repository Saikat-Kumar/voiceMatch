import yt_dlp as youtube_dl
import subprocess
import matplotlib.pyplot as plt
import soundfile as sf
import numpy as np
def download_audio(url, output_file):
    ydl_opts = {
        'format': 'bestaudio/best',
        'outtmpl': output_file + '.%(ext)s',
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'wav',
            'preferredquality': '192',
        }],
        'ignoreerrors': True,  # Add this option to ignore errors
    }
    with youtube_dl.YoutubeDL(ydl_opts) as ydl:
        ydl.download([url])



# URL of the YouTube video
youtube_url = 'https://www.youtube.com/watch?v=I49VNQ6lmKk'


output_file = 'download'
# Download audio from YouTube
download_audio(youtube_url, output_file)
signal, fs = sf.read('download.wav')
print(f'Audio saved as {output_file}')
Time = np.linspace(0, len(signal) / fs, num=len(signal))

color="tab:blue"
start_time = 0 / fs
end_time = start_time + (len(signal) / fs)

plt.plot(np.linspace(start_time, end_time, len(signal)), signal, color=color)
plt.xlabel("Time (s)")
plt.xlim([start_time, end_time])

max_amp = np.max(np.abs([np.max(signal), np.min(signal)]))
plt.ylim([-max_amp, max_amp])

plt.tight_layout()
plt.show()
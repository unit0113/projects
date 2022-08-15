import wave
import matplotlib.pyplot as plt
import numpy as np

CHUNK = 2048

wf = wave.open('Python_Playground\song_visualizer\Christopher Tin - The Storm-Driven Sea.wav', 'rb')

sample_freq = wf.getframerate()
num_samples = wf.getnframes()
signal_wave = wf.readframes(-1)

wf.close()

time_audio = num_samples / sample_freq

signal_array = np.frombuffer(signal_wave, dtype=np.int32)
ft = np.abs(np.fft.fft(signal_array))
times = np.linspace(0, time_audio, num=num_samples)
freq = np.linspace(0, sample_freq, len(ft))
f_bins = int(len(ft) * 0.1)

plt.figure(figsize=(15, 5))
#plt.plot(times, signal_array)
plt.plot(freq[:f_bins], ft[:f_bins])
plt.title("Audio Signal")
plt.ylabel("Signal Wave")
plt.xlabel("Time")
#plt.xlim(0,time_audio)

plt.show()
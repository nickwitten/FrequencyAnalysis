import os
import sys
import numpy as np
from matplotlib import pyplot as plt
from scipy.io import wavfile
from note_freqs import note_freqs
import re


def plot_time(y, sample_rate, ax):
    n = y.size
    x = np.linspace(0, n*sample_rate, num=n, endpoint=None)
    ax.plot(x, y);

def plot_freq(y, sample_rate, ax):
    yf = np.fft.rfft(y)
    xf = np.fft.rfftfreq(y.size, 1 / sample_rate)
    range_mask = (xf >= note_freqs['A0']) & (xf <= note_freqs['C8'])
    ax.semilogy()
    ax.plot(xf[range_mask], np.abs(yf[range_mask]))
    # print(np.std(yf[range_mask]))
    # ax.set_xlim(note_freqs['A0'], note_freqs['C8'])

freqs = np.array(list(note_freqs.values()))
freq_diffs = np.diff(freqs)
def plot_piano(y, sample_rate, ax):
    yf = np.fft.rfft(y)
    yf = np.abs(yf)  # Get magnitude (intensity)
    xf = np.fft.rfftfreq(y.size, 1 / sample_rate)
    intensities = {}
    for i, (note, freq) in enumerate(note_freqs.items()):
        # Only include notes on the piano
        if freq <= note_freqs['A0'] or freq >= note_freqs['C8']:
            continue
        # Chose the frequency range to average over for this note
        bucket_size = freq_diffs[i-1]
        bucket_mask = (xf >= (freq - bucket_size/2)) & (xf <= (freq + bucket_size/2))
        intensity = np.mean(yf[bucket_mask])
        intensities[re.sub("/.*", "", note)] = intensity
    max_intensity = np.array(list(intensities.values())).max()
    for note, intensity in intensities.items():
        intensities[note] = intensity / max_intensity
    print(np.std(list(intensities.values())))
    ax.bar(list(intensities.keys()), list(intensities.values()))


# Get impulse wav files
impulse_dir = '.\\impulses_cut\\'
fns = os.listdir(impulse_dir)
# Setup plotting figures
ncols = 2
nrows = len(fns) // ncols + (len(fns) % ncols)
fig1 = plt.figure(tight_layout=True)
fig2 = plt.figure(tight_layout=True)
fig3 = plt.figure(tight_layout=True)
# Loop through and plot the signals in the time and frequency domains
for i, f in enumerate(fns):
    ax1 = fig1.add_subplot(nrows, ncols, i+1)
    ax2 = fig2.add_subplot(nrows, ncols, i+1)
    ax3 = fig3.add_subplot(nrows, ncols, i+1)
    ax1.set_title(f)
    ax2.set_title(f)
    ax3.set_title(f)

    sample_rate, y = wavfile.read(os.path.join(impulse_dir, f), mmap=False)
    # scale y up to int max
    dtype = y.dtype
    y = y * (np.iinfo(dtype).max / y.max())
    y = y.astype(dtype)

    plot_time(y, sample_rate, ax1)
    plot_freq(y, sample_rate, ax2)
    plot_piano(y, sample_rate, ax3)


plt.show()

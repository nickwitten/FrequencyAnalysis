import os
import sys
import numpy as np
from matplotlib import pyplot as plt
from scipy.io import wavfile
from note_freqs import note_freqs
import re


def plot_time(y, sample_rate, ax):
    y = cut_impulse(y)
    c80 = clarity(y, sample_rate)
    x = np.linspace(0, y.size * 1 / sample_rate, num=y.size, endpoint=None)
    y = y / np.iinfo(y.dtype).max
    cutoff = int(0.08 / (1 / sample_rate))
    ax.plot(x[:cutoff], y[:cutoff], label="First 80 ms", color="Green")
    ax.plot(x[cutoff:], y[cutoff:], label="Late Energy", color="Red")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Amplitude")
    ax.legend()
    ax.set_title(ax.title.get_text() + f' C80: {round(c80, 2)} dB')

def plot_time_db(y, sample_rate, ax):
    y = y.astype(np.int32)
    y = np.abs(cut_impulse(y, end=False))
    chunksize = 300
    numchunks = y.size // chunksize
    y = y[:numchunks*chunksize]
    chunks = np.array_split(y, numchunks)
    y = np.array([20*np.log10( np.sqrt(np.mean(chunk**2)) ) for chunk in chunks])
    t20, t30, t60 = reverb(y, sample_rate / chunksize)
    x = np.linspace(0, numchunks * chunksize * 1 / sample_rate, num=numchunks, endpoint=None)
    ax.plot(x, y)
    ax.axvline(t20, label="T20", color="green")
    ax.axvline(t30, label="T30", color="purple")
    ax.axvline(t60, label="T60", color="red")
    ax.set_ylabel("Amplitude (dB)")
    ax.set_xlabel("Time (s)")
    ax.legend()
    ax.set_title(ax.title.get_text() + f' T60: {round(t60, 2)} s')


def plot_freq(y, sample_rate, ax, log=False):
    yf = np.fft.rfft(y)
    xf = np.fft.rfftfreq(y.size, 1 / sample_rate)
    range_mask = (xf >= note_freqs['A0']) & (xf <= note_freqs['C8'])
    if log:
        yf = 20 * np.log10(np.abs(yf))
    ax.plot(xf[range_mask], np.abs(yf[range_mask]))
    ax.set_title(ax.title.get_text() + f'STD: {round(np.std(np.abs(yf)), 2)}')

freqs = np.array(list(note_freqs.values()))
freq_diffs = np.diff(freqs)
def plot_piano(y, sample_rate, ax, log=False):
    yf = np.fft.rfft(y)
    yf = np.abs(yf)  # Get magnitude (intensity)
    xf = np.fft.rfftfreq(y.size, 1 / sample_rate)
    intensities = list()
    labels = list()
    for i, (note, freq) in enumerate(note_freqs.items()):
        # Only include notes on the piano
        if freq <= note_freqs['A0'] or freq >= note_freqs['C8']:
            continue
        # Chose the frequency range to average over for this note
        bucket_size = freq_diffs[i-1]
        bucket_mask = (xf >= (freq - bucket_size/2)) & (xf <= (freq + bucket_size/2))
        intensities.append(np.mean(yf[bucket_mask]))
        labels.append(re.sub("/.*", "", note))
    intensities = np.array(intensities)
    intensities = intensities / intensities.max()  # Normalize
    if log:
        intensities = 20 * np.log10(intensities)
    ax.bar(labels, intensities)
    ax.set_title(ax.title.get_text() + f' STD: {round(np.std(intensities), 2)}')

def reverb(y, sample_rate):
    inds = np.argwhere(y < y[0] - 30).flatten()
    inds = inds[inds*1/sample_rate > 0.1]
    t30 = 1 / sample_rate * inds[0]

    inds = np.argwhere(y < y[0] - 20).flatten()
    inds = inds[inds*1/sample_rate > 0.1]
    t20 = 1 / sample_rate * inds[0]

    return t20, t30, np.mean([t30 * 2, t20 * 3])

def clarity(y, sample_rate):
    front_80 = int(0.08 * sample_rate)
    fenergy = np.sum(np.square(y[:front_80] * (1 / sample_rate)))
    benergy = np.sum(np.square(y[front_80:] * (1 / sample_rate)))
    c80 = 10 * np.log10(fenergy / benergy)
    return c80

def cut_impulse(y, end=True):
    # First we need to cut out the signal to
    # exactly when the impulse response starts
    # and stops.
    # We can find when the impulse starts by
    # finding the largest jump in amplitude
    start = np.argmax(np.diff(np.abs(y)))
    # We need to find when reverberation stops
    # and noise begins, use the last 1000 sample
    # max to estimate the noise floor
    noise_floor = np.max(np.abs(y[-1000:]))
    # find the first time that the average in
    # a sliding window is less than the noise
    # threshold
    cutoff = 0
    win_size = 100
    for i in range(int(0.08 * sample_rate), y.size):
        if np.mean(np.abs(y[i:i+win_size])) <= noise_floor:
            cutoff = i
            break
    if cutoff == 0:
        raise Exception('End of impulse not found')
    # Now we can cut out our impulse response
    # Must promote to int32 because we will square it
    if not end:
        return y[start:]
    return y[start:cutoff]

# Get impulse wav files
impulse_dir = '.\\impulses\\'
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
    ax1.set_title(re.sub("\..*", "", f))
    ax2.set_title(re.sub("\..*", "", f))
    ax3.set_title(re.sub("\..*", "", f))

    sample_rate, y = wavfile.read(os.path.join(impulse_dir, f), mmap=False)

    plot_time(np.ndarray.copy(y), sample_rate, ax1)
    plot_time_db(np.ndarray.copy(y), sample_rate, ax2)
    plot_freq(np.copy(y), sample_rate, ax3, log=True)
    # plot_piano(y, sample_rate, ax3)
    # plot_piano(y, sample_rate, ax3, log=True)


plt.show()

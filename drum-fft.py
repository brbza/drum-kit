import numpy as np
from matplotlib import pyplot as plt
from scipy.fft import rfft, rfftfreq
from scipy.io import wavfile

drums = [
    "floor",
    "kick",
    "snare",
    "tom-1",
    "tom-2",
    "full-kit-3-mics",
    "full-kit-7-mics"
]

harmonic_colours = [
    "red",
    "orange",
    "yellow",
    "green",
    "blue",
    "purple",
    "brown",
    "black",
]

for drum in drums:
    # read the .wav file
    sample_rate, yt = wavfile.read(f'audio/{drum}.wav')

    # calculate fourier transform of one of the channels
    yf = rfft(yt.T[0])
    xf = rfftfreq(len(yt.T[0]), 1 / sample_rate)

    # compute the fundamental frequency
    fundamental = xf[np.argmax(yf)]

    # compute the index at 1 KHz to zoom the plots
    _1KHz_index = int((len(xf)*1000)/(sample_rate/2))

    plt.plot(
        xf[:_1KHz_index],
        np.abs(yf[:_1KHz_index]),
        label="spectrum",
        color="grey"
    )

    for idx, colour in enumerate(harmonic_colours):
        harmonic_freq = (idx+1) * fundamental
        plt.axvline(
            x=harmonic_freq,
            label=f'{harmonic_freq:.2f} Hz',
            color=colour
        )

    plt.xlabel("Frequency")
    plt.ylabel("Power")
    plt.title(f"{drum.capitalize()} Drum")
    plt.legend()
    plt.savefig(f"images/{drum}.png")
    plt.close()

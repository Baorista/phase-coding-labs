#!/usr/bin/env python3
import numpy as np
import scipy.io.wavfile as wavfile

sr = 44100
t = np.arange(sr * 2) / sr

music = (SEG_AMP_1 * np.sin(2*np.pi*440*t) +
         SEG_AMP_2 * np.sin(2*np.pi*880*t) +
         500 * np.random.randn(len(t)))
wavfile.write("input/music.wav", sr, music.astype(np.int16))

speech = (5000 * np.sin(2*np.pi*300*t) * np.sin(2*np.pi*3*t)).astype(np.int16)
wavfile.write("input/speech.wav", sr, speech)

noise = (3000 * np.random.randn(len(t))).astype(np.int16)
wavfile.write("input/noise.wav", sr, noise)

print("Created 3 WAV files")

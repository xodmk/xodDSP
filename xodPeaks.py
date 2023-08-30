# -*- coding: utf-8 -*-
# *****************************************************************************
# /////////////////////////////////////////////////////////////////////////////
# \\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\
# *****************************************************************************
#
# __::((xodPeaks.py))::__
#
# XODMK Python Xperimental Peak Detection
# required lib:
#
#
# Requirements
# sudo apt-get install python3-tk
#
# *****************************************************************************
# /////////////////////////////////////////////////////////////////////////////
# \\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\
# *****************************************************************************

import os
import sys
import numpy as np
import librosa
import librosa.display
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

# import xodClocks as clks
# import xodWavGen as wavGen


# // *---------------------------------------------------------------------* //
# // *---------------------------------------------------------------------* //

# assumes python projects are located in xodPython

currentDir = os.getcwd()
rootDir = os.path.dirname(currentDir)
audioSrcDir = rootDir + "/data/src/wav"
audioOutDir = rootDir + "/data/res/wavout"

print("rootDir: " + rootDir)
print("currentDir: " + currentDir)
print("audioSrcDir: " + audioSrcDir)
print("audioOutDir: " + audioOutDir)

sys.path.insert(0, rootDir+'/xodma')

from xodmaAudioTools import load_wav, write_wav, valid_audio, resample, peak_pick
from xodmaOnset import detectOnset, onset_strength
from xodmaSpectralUtil import frames_to_time
from xodmaSpectralPlot import specshow


sys.path.insert(1, rootDir+'/xodUtil')
import xodPlotUtil as xodplt


# temp python debugger - use >>>pdb.set_trace() to set break
import pdb

# // *---------------------------------------------------------------------* //

plt.close('all')

# // *---------------------------------------------------------------------* //
# // *--User Settings - Primary parameters--*
# // *---------------------------------------------------------------------* //

wavSrc = 'The_Amen_Break_48K.wav'

# length of input signal:
# '0'   => full length of input .wav file
# '###' => usr defined length in SECONDS
wavLength = 0


# // *---------------------------------------------------------------------* //
# // *---------------------------------------------------------------------* //

# inputs:  wavIn, audioSrcDir, wavLength
# outputs: ySrc_ch1, ySrc_ch2, numChannels, fs, ySamples

# Load Stereo .wav file

audioSrc = audioSrcDir + '/' + wavSrc

[aSrc, aNumChannels, afs, aLength, aSamples] = load_wav(audioSrc, wavLength, True)

if aNumChannels == 2:
    aSrc_ch1 = aSrc[:, 0]
    aSrc_ch2 = aSrc[:, 1]
else:
    aSrc_ch1 = aSrc
    aSrc_ch2 = 0

sr = afs
num_samples = aSrc_ch1.size

# // *--- Plot - source signal ---*

if 1:

    plt.figure(figsize=(14, 5))
    librosa.display.waveshow(aSrc_ch1, sr=sr)

    fnum = 3
    pltTitle = 'Input Signals: aSrc_ch1'
    pltXlabel = 'sinArray time-domain wav'
    pltYlabel = 'Magnitude'

    # define a linear space from 0 to 1/2 Fs for x-axis:
    xaxis = np.linspace(0, len(aSrc_ch1), len(aSrc_ch1))

    xodplt.xodPlot1D(fnum, aSrc_ch1, xaxis, pltTitle, pltXlabel, pltYlabel)

# pdb.set_trace()

# // *---------------------------------------------------------------------* //
# // *---------------------------------------------------------------------* //

# Compute an onset envelope:
hop_length = 256

onset_envelope = librosa.onset.onset_strength(aSrc_ch1, sr=sr, hop_length=hop_length)
print("onset_envelope.shape = " + str(onset_envelope.shape))
print("onset_envelope = " + str(onset_envelope))

onset_envelope_xodma = onset_strength(aSrc_ch1, sr=sr, hop_length=hop_length)
print("onset_envelope_xodma.shape = " + str(onset_envelope_xodma.shape))
print("onset_envelope_xodma = " + str(onset_envelope_xodma))


# // *---------------------------------------------------------------------* //
# // *----- Librosa Peak Pick -----*

# Get the frame indices of the peaks:
peaks_librosa = librosa.util.peak_pick(onset_envelope, 7, 7, 7, 7, 0.5, 5)

# peaks_xodma = peak_pick(onset_envelope, 7, 7, 7, 7, 0.5, 5)
peaks_xodma = peak_pick(onset_envelope_xodma, 7, 7, 7, 7, 0.5, 5)

# // *---------------------------------------------------------------------* //
# // *----- Plotting Peak Pick -----*

N = len(aSrc_ch1)
T = N / float(sr)
t = np.linspace(0, T, len(onset_envelope))

pdb.set_trace()

print("peaks_librosa = " + str(peaks_librosa))
plt.figure(figsize=(14, 5))
plt.plot(t, onset_envelope)
plt.grid(False)
plt.vlines(t[peaks_librosa], 0, onset_envelope.max(), color='r', alpha=0.7)
plt.title('peaks_librosa')
plt.xlabel('Time (sec)')
plt.xlim(0, T)
plt.ylim(0)

print("peaks_xodma = " + str(peaks_xodma))
plt.figure(figsize=(14, 5))
plt.plot(t, onset_envelope_xodma)
plt.grid(False)
plt.vlines(t[peaks_xodma], 0, onset_envelope_xodma.max(), color='r', alpha=0.7)
plt.title('peaks_xodma')
plt.xlabel('Time (sec)')
plt.xlim(0, T)
plt.ylim(0)

# // *---------------------------------------------------------------------* //
# // *----- XODMA onset detection -----*

hop = hop_length

# Matches: peaks = peak_pick(onset_env, 7, 7, 7, 7, 0.5, 5)
peakThresh = 0.5
peakWait = 5


def xodmaPeaks(wavIn, sr, hop, peakThresh, peakWait, **kwargs):
    """
    Wrapper for XODMA peak_pick function (modified clone of librosa func)
    wavIn      : input 1D time domain signal (audio, signal, envelope, etc.)
    sr         : sample rate (normalized for STFT 48KHz audio)
    hop        : STFT Hop length (normalized for hop length 256 (?!))
    peakThresh : threshold offset above windowed-mean to qualify a peak
    peakWait   : number of samples to wait after picking a peak

    ** FIXIT FIXIT - good results, but nonsensical parameterization...
    Current Example:
    # Matches: peaks = peak_pick(onset_env, 7, 7, 7, 7, 0.5, 5)  # -> default librosa params
    # >> peakThresh = 0.5
    # >> peakWait = 5
    """

    kwargs.setdefault('pre_max', 0.04 * sr // hop)      # 7.0
    kwargs.setdefault('post_max', 0.04 * sr // hop)     # 7.0
    kwargs.setdefault('pre_avg', 0.04 * sr // hop)      # 7.0
    kwargs.setdefault('post_avg', 0.04 * sr // hop)     # 7.0
    kwargs.setdefault('delta', peakThresh)
    kwargs.setdefault('wait', peakWait)                 # 30ms

    peaks = peak_pick(wavIn, **kwargs)

    return peaks


xodmaPeaks_res = xodmaPeaks(onset_envelope, sr, hop, peakThresh, peakWait)

print("xodmaPeaks_res = " + str(xodmaPeaks_res))

plt.figure(figsize=(14, 5))
plt.plot(t, onset_envelope)
plt.grid(False)
plt.vlines(t[xodmaPeaks_res], 0, onset_envelope.max(), color='r', alpha=0.7)
plt.title('xodmaPeaks_res')
plt.xlabel('Time (sec)')
plt.xlim(0, T)
plt.ylim(0)

# // *---------------------------------------------------------------------* //

plt.show()

print('\n')
print('// *--------------------------------------------------------------* //')
print('// *---::done::---*')
print('// *--------------------------------------------------------------* //')
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
audioSrcDir = rootDir + "/pyAudio/wavsrc/"
audioOutDir = rootDir + "/pyAudio/wavout/"

print("rootDir: " + rootDir)
print("currentDir: " + currentDir)

sys.path.insert(0, rootDir+'/xodma')

from xodmaAudioTools import load_wav, write_wav, valid_audio, resample
# from xodmaAudioTools import samples_to_time, time_to_samples, fix_length
# from xodmaSpectralTools import amplitude_to_db, stft, istft, peak_pick
# from xodmaSpectralTools import magphase
# from xodmaSpectralUtil import frames_to_time
# from xodmaSpectralPlot import specshow

sys.path.insert(1, rootDir+'/xodUtil')
import xodPlotUtil as xodplt


# temp python debugger - use >>>pdb.set_trace() to set break
import pdb

# // *---------------------------------------------------------------------* //

plt.close('all')

# // *---------------------------------------------------------------------* //
# // *--User Settings - Primary parameters--*
# // *---------------------------------------------------------------------* //

wavSrcA = 'slothForest_btx01.wav'
# wavSrcB = 'scoolreaktor_beatx03.wav'
# wavSrcB = 'gorgulans_beatx01.wav'

# length of input signal:
# '0'   => full length of input .wav file
# '###' => usr defined length in SECONDS
wavLength = 0


# // *---------------------------------------------------------------------* //
# // *---------------------------------------------------------------------* //

# inputs:  wavIn, audioSrcDir, wavLength
# outputs: ySrc_ch1, ySrc_ch2, numChannels, fs, ySamples

# Load Stereo .wav file

audioSrcA = audioSrcDir + wavSrcA
# audioSrcB = audioSrcDir+wavSrcB

[aSrc, aNumChannels, afs, aLength, aSamples] = load_wav(audioSrcA, wavLength, True)
# [bSrc, bNumChannels, bfs, bLength, bSamples] = load_wav(audioSrcB, wavLength, True)

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

N = len(aSrc_ch1)
T = N / float(sr)
t = np.linspace(0, T, len(onset_envelope))

# Get the frame indices of the peaks:
onset_frames = librosa.util.peak_pick(onset_envelope, 7, 7, 7, 7, 0.5, 5)

print("onset_frames = " + str(onset_frames))

plt.figure(figsize=(14, 5))
plt.plot(t, onset_envelope)
plt.grid(False)
plt.vlines(t[onset_frames], 0, onset_envelope.max(), color='r', alpha=0.7)
plt.xlabel('Time (sec)')
plt.xlim(0, T)
plt.ylim(0)

# // *---------------------------------------------------------------------* //

plt.show()

print('\n')
print('// *--------------------------------------------------------------* //')
print('// *---::done::---*')
print('// *--------------------------------------------------------------* //')
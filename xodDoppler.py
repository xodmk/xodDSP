# -*- coding: utf-8 -*-
# *****************************************************************************
# /////////////////////////////////////////////////////////////////////////////
# \\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\
# *****************************************************************************
#
# __::((xodDoppler.py))::__
#
# XODMK Python Xperimental Doppler Processor
# required lib:
# odmkClocks ; xodWavGen
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
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

import xodClocks as clks
import xodWavGen as wavGen


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
# // *---------------------------------------------------------------------* //

# // *---------------------------------------------------------------------* //
# // *--User Settings - Primary parameters--*
# // *---------------------------------------------------------------------* //

# srcSel: 0 = wavSrc, 1 = amenBreak, 2 = sineWave48K,
#         3 = multiSin test, 4 = text array input

srcSel = 0

# STEREO source signal
# wavSrc = 'The_Amen_Break_odmk.wav'

# MONO source signal
# wavSrc = 'multiSinOut48KHz_1K_3K_5K_7K_9K_16sec.wav'

wavSrcA = 'slothForest_btx01.wav'
# wavSrcB = 'scoolreaktor_beatx03.wav'
# wavSrcB = 'gorgulans_beatx01.wav'

# length of input signal:
# '0'   => full length of input .wav file
# '###' => usr defined length in SECONDS
wavLength = 0

NFFT = 2048
STFTHOP = int(NFFT / 4)
WIN = 'hann'

''' Valid Window Types: 

boxcar
triang
blackman
hamming
hann
bartlett
flattop
parzen
bohman
blackmanharris
nuttall
barthann
kaiser (needs beta)
gaussian (needs standard deviation)
general_gaussian (needs power, width)
slepian (needs width)
dpss (needs normalized half-bandwidth)
chebwin (needs attenuation)
exponential (needs decay scale)
tukey (needs taper fraction)

'''

# // *---------------------------------------------------------------------* //
# // *---------------------------------------------------------------------* //

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

NFFT = 2048
STFTHOP = int(NFFT/4)
WIN = 'hann'


''' Valid Window Types: 

boxcar
triang
blackman
hamming
hann
bartlett
flattop
parzen
bohman
blackmanharris
nuttall
barthann
kaiser (needs beta)
gaussian (needs standard deviation)
general_gaussian (needs power, width)
slepian (needs width)
dpss (needs normalized half-bandwidth)
chebwin (needs attenuation)
exponential (needs decay scale)
tukey (needs taper fraction)

'''

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

# if bNumChannels == 2:
#    bSrc_ch1 = bSrc[:,0];
#    bSrc_ch2 = bSrc[:,1];
# else:
#    bSrc_ch1 = bSrc;
#    bSrc_ch2 = 0;


# aT = 1.0 / afs
# print('\nsample period: ------------------------- '+str(aT))
# print('wav file datatype: '+str(sf.info(audioSrcA).subtype))


# // *--- Plot - source signal ---*

if 1:
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

fs = afs
# fs = 48000
# ch1 = (np.sin(2*np.pi*np.arange(80000)*2000/fs)).astype(np.float32)

num_samples = aSrc_ch1.size

# force num_samples to be even
num_samples = 2 * int(num_samples / 2)


V = 330
A = -6
B = 6
C = 1
X = np.arange(A, B, (B - A) / num_samples)
vels = 60 * X / ((C**2 + X**2)**0.5)
f = (V - vels) / V

# plt.plot(f)
# plt.show()

doppler = np.zeros(num_samples)
delta = 1.0 / fs
index = 0
indices = (1 / f)
for i in range(num_samples):
	value = aSrc_ch1[i]
	if index * fs >= num_samples - 1:
		break
	pos = index * fs
	mod = pos % 1.0
	doppler[int(pos)] += value * (1 - mod)
	doppler[int(pos) + 1] += value * mod
	index += delta * indices[i]
N = doppler.size

#pdb.set_trace()

# apply amplitude fade in and fade out (it's linear, TODO: inverse square)
doppler = doppler * np.concatenate((np.arange(N/2), np.arange(N/2, 0, -1))) / (N/2)

# normalize signal scale to -1, 1
doppler = -1 + (2 * (doppler - np.min(doppler)) / (np.max(doppler) - np.min(doppler)))


dopplerWavRes1Out = audioOutDir + 'dopplerWavRes1.wav'
write_wav(dopplerWavRes1Out, doppler, fs)

unmoddedWavOut = audioOutDir + 'unmoddedWav.wav'
write_wav(unmoddedWavOut, aSrc_ch1, fs)

plt.show()
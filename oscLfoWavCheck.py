# -*- coding: utf-8 -*-
# *****************************************************************************
# /////////////////////////////////////////////////////////////////////////////
# \\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\
# *****************************************************************************
#
# __::((oscLfoWavCheck.py))::__
#
# Python test scriopt for verification & plotting Osc/Lfo/Wavforms
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
import scipy as sp
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
audioOutDir = currentDir + "audio/wavout/"

print("currentDir: " + currentDir)
print("rootDir: " + rootDir)


sys.path.insert(0, rootDir+'/xodUtil')
import xodPlotUtil as xodplt

sys.path.insert(1, rootDir+'/xodma')
from xodmaAudioTools import write_wav


# temp python debugger - use >>>pdb.set_trace() to set break
import pdb

# run this command to de-embed plots
# %matplotlib qt

# // *---------------------------------------------------------------------* //

print('// //////////////////////////////////////////////////////////////// //')
print('// *--------------------------------------------------------------* //')
print('// *---::XODMK Waveform Generator 1::---*')
print('// *--------------------------------------------------------------* //')
print('// \\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\ //')


# \\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\
# // *---------------------------------------------------------------------* //
# // *--Math Functions--*
# // *---------------------------------------------------------------------* //

def cyclicZn(n):
    """ calculates the Zn roots of unity """
    cZn = np.zeros((n, 1))*(0+0j)    # column vector of zero complex values
    for k in range(n):
        # z(k) = e^(((k)*2*pi*1j)/n)        # Define cyclic group Zn points
        cZn[k] = np.cos((k * 2 * np.pi) / n) + np.sin((k * 2 * np.pi) / n) * 1j   # Euler's identity

    return cZn


# /////////////////////////////////////////////////////////////////////////////
# #############################################################################
# begin Test:
# #############################################################################
# \\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\

print('\n')
print('// *--------------------------------------------------------------* //')
print('// *---::Set Parameters for test output::---*')
print('// *--------------------------------------------------------------* //')


# length of x in seconds:
xLength = 5
# audio sample rate:
sr = 48000
# sample period
T = 1.0 / sr

# audio sample bit width
bWidth = 24


bpm = 133.0
# time signature: 0 = 4/4; 1 = 3/4
timeSig = 0
# video frames per second:
framesPerSec = 30.0


# Write waveform to .wav file
createWavFile = 0


# test freq
testFreq1 = 777.0
testPhase1 = 0


# select generated waveforms
genMonoSin = 0
genMonoTri = 0
genLFO = 0
genSinArray = 0
genOrthoSinArray = 0
genCompositeSinArray = 0

genWavetableOsc = 1
if genWavetableOsc == 1:
    # shape:
    # 1=sin,     2=cos,     3=tri,     4=saw-up,    5=saw-dn,
    # 6=exp-up,  7=exp-dn,  8=log-up,  9=log-dn,    10=cheby,
    # 11=pulse1, 12=pulse2, 13=pulse3, 14=pulse4,   15=user
    shape = 'pulse3'

genPWMOsc = 0


# /////////////////////////////////////////////////////////////////////////////
# #############################################################################
# begin : object definition
# #############################################################################
# \\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\


print('\n')
print('// *--------------------------------------------------------------* //')
print('// *---::Instantiate clock & signal Generator objects::---*')
print('// *--------------------------------------------------------------* //')

tbClocks = clks.xodClocks(xLength, sr, bpm, framesPerSec)

numSamples = tbClocks.totalSamples


wavGenOutDir = audioOutDir+'wavGenOutDir/'

tbWavGen = wavGen.xodWavGen(sr, xLength, wavGenOutDir)


# // *---------------------------------------------------------------------* //

tbclkDownBeats = tbClocks.clkDownBeats()


# /////////////////////////////////////////////////////////////////////////////
# #############################################################################
# begin : Read .txt files from xodLFO HLS function
# #############################################################################
# \\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\

xodLFO_resDir = currentDir + "/home/eschei/xodmk/xodCode/xodHLS/audio/data/output/"

xodLFO_sin_src = xodLFO_resDir + "dds_lfoSin_out.txt"
xodLFO_sqr_src = xodLFO_resDir + "dds_lfoSqr_out.txt"
xodLFO_saw_src = xodLFO_resDir + "dds_lfoSaw_out.txt"
xodLFO_soc_src = xodLFO_resDir + "dds_startOfCycle_out.txt"


# // *---------------------------------------------------------------------* //    
    
if plotSinArray == 1:
    
    # // *---------------------------------------------------------------------* //   
    # // *---Array of Sines wave plots---*
    
    # Test FFT length
    N = 4096
    
    tLen = N

    numFreqs = numFreqSinArray    # defined above for gen of sinArray

    yArray = np.array([])
    yScaleArray = np.array([])
    # for h in range(len(sinArray[0, :])):
    for h in range(numFreqs):    
        yFFT = sp.fft(sinArray[h, 0:N])
        yArray = np.concatenate((yArray, yFFT))
        yScaleArray = np.concatenate((yScaleArray, 2.0/N * np.abs(yFFT[0:int(N/2)])))
    yArray = yArray.reshape((numFreqs, N))
    yScaleArray = yScaleArray.reshape((numFreqs, int(N/2)))

    fnum = 30
    pltTitle = 'Input Signals: sinArray (first '+str(tLen)+' samples)'
    pltXlabel = 'sinArray time-domain wav'
    pltYlabel = 'Magnitude'
    
    # define a linear space from 0 to 1/2 Fs for x-axis:
    xaxis = np.linspace(0, tLen, tLen)
    
    # pdb.set_trace()
    
    xodplt.xodMultiPlot1D(fnum, sinArray, xaxis, pltTitle, pltXlabel, pltYlabel, colorMap='hsv')

    fnum = 31
    pltTitle = 'FFT Mag: yScaleArray multi-osc '
    pltXlabel = 'Frequency: 0 - '+str(sr / 2)+' Hz'
    pltYlabel = 'Magnitude (scaled by 2/N)'
    
    # define a linear space from 0 to 1/2 Fs for x-axis:
    xfnyq = np.linspace(0.0, 1.0/(2.0*T), int(N/2))
    
    xodplt.xodMultiPlot1D(fnum, yScaleArray, xfnyq, pltTitle, pltXlabel, pltYlabel, colorMap='hsv')
    

# // *---------------------------------------------------------------------* //

if plotOrthoSinArray == 1:
    
    # // *---------------------------------------------------------------------* //   
    # // *---Array of Orthogonal Sines wave plots---*
    
    # Test FFT length
    N = 4096
    
    tLen = N
    
    numFreqs = numOrthoFreq

    yOrthoArray = np.array([])
    yOrthoScaleArray = np.array([])
    # for h in range(len(sinArray[0, :])):
    for h in range(numFreqs):
        yOrthoFFT = sp.fft(orthoSinArray[h, 0:N])
        yOrthoArray = np.concatenate((yOrthoArray, yOrthoFFT))
        yOrthoScaleArray = np.concatenate((yOrthoScaleArray, 2.0/N * np.abs(yOrthoFFT[0:int(N/2)])))
    yOrthoArray = yOrthoArray.reshape((numFreqs, N))
    yOrthoScaleArray = yOrthoScaleArray.reshape(numFreqs, (int(N/2)))

    fnum = 32
    pltTitle = 'Input Signals: orthoSinArray (first '+str(tLen)+' samples)'
    pltXlabel = 'orthoSinArray time-domain wav'
    pltYlabel = 'Magnitude'
    
    # define a linear space from 0 to 1/2 Fs for x-axis:
    xaxis = np.linspace(0, tLen, tLen)
    
    xodplt.xodMultiPlot1D(fnum, orthoSinArray, xaxis, pltTitle, pltXlabel, pltYlabel, colorMap='hsv')

    fnum = 33
    pltTitle = 'FFT Mag: yOrthoScaleArray multi-osc '
    pltXlabel = 'Frequency: 0 - '+str(sr / 2)+' Hz'
    pltYlabel = 'Magnitude (scaled by 2/N)'
    
    # define a linear space from 0 to 1/2 Fs for x-axis:
    xfnyq = np.linspace(0.0, 1.0/(2.0*T), N/2)
    
    xodplt.xodMultiPlot1D(fnum, yOrthoScaleArray, xfnyq, pltTitle, pltXlabel, pltYlabel, colorMap='hsv')
   
# // *---------------------------------------------------------------------* //

if plotCompositeSinArray == 1:

    # // *---------------------------------------------------------------------* //
    # // *---Composit Sines wave plots---*

    # Test FFT length
    N = 4096

    # odmkTestFreqArray5_3
    tLen = N
    
    sig = multiSinOut[0:tLen]
    # define a linear space from 0 to 1/2 Fs for x-axis:
    xaxis = np.linspace(0, tLen, tLen)

    fnum = 40
    pltTitle = 'SigGen output: sinOrth5Comp1 Composite waveform (first '+str(tLen)+' samples)'
    pltXlabel = 'time'
    pltYlabel = 'Magnitude (scaled)'
    
    xodplt.xodPlot1D(fnum, sig, xaxis, pltTitle, pltXlabel, pltYlabel)    

    # // *-----------------------------------------------------------------* //

    ySinComp1 = multiSinOut[0:N]

    sinComp1_FFT = sp.fft(ySinComp1)
    sinComp1_FFTscale = 2.0/N * np.abs(sinComp1_FFT[0:int(N/2)])

    fnum = 41
    pltTitle = 'FFT Mag: sinComp1_FFTscale Composite waveform'
    pltXlabel = 'Frequency: 0 - '+str(sr / 2)+' Hz'
    pltYlabel = 'Magnitude (scaled by 2/N)'
    
    # sig <= direct
    
    # define a linear space from 0 to 1/2 Fs for x-axis:
    xfnyq = np.linspace(0.0, 1.0/(2.0*T), int(N/2))
    
    xodplt.xodPlot1D(fnum, sinComp1_FFTscale, xfnyq, pltTitle, pltXlabel, pltYlabel)    

    # // *-----------------------------------------------------------------* //


# // *---------------------------------------------------------------------* //

if plotWavetableOsc == 1:
    
    # // *---------------------------------------------------------------------* //
    # // *---Mono Sin/Cos plots---*
    
    # FFT length
    N = 2048

    y = xodOsc[0:N]

    # pdb.set_trace()

    # forward FFT
    y_FFT = np.fft.fft(y)
    
    # y_Mag = np.abs(y_FFT)
    # y_Phase = np.arctan2(y_FFT.imag, y_FFT.real)
    
    # scale and format FFT out for plotting
    y_FFTscale = 2.0/N * np.abs(y_FFT[0:int(N/2)])

    # define a sub-range for wave plot visibility
    tLen = 500
    
    # Plot y1 time domain:
    fnum = 50
    pltTitle = 'WaveTable OSC '+str(freqCtrl)+' Hz, (first '+str(tLen)+' samples)'
    pltXlabel = 'time'
    pltYlabel = 'Magnitude'

    sig = y[0:tLen]
    # define a linear space from 0 to 1/2 Fs for x-axis:
    xaxis = np.linspace(0, tLen, tLen)

    xodplt.xodPlot1D(fnum, sig, xaxis, pltTitle, pltXlabel, pltYlabel)

    # sig2 = odmkOsc2[0:tLen]
    # sigN = np.concatenate((sig, sig2))
    # sigN = sigN.reshape((2, tLen))
    #
    # odmkplt.odmkMultiPlot1D(fnum, sigN, xaxis, pltTitle, pltXlabel, pltYlabel)

    # plot y freq domain:
    fnum = 51
    pltTitle = 'FFT Mag WaveTable OSC '+str(freqCtrl)+' Hz'
    pltXlabel = 'Frequency: 0 - '+str(sr / 2)+' Hz'
    pltYlabel = 'Magnitude (scaled by 2/N)'
    
    # define a linear space from 0 to 1/2 Fs for x-axis:
    xfnyq = np.linspace(0.0, 1.0/(2.0*T), int(N/2))
    
    xodplt.xodPlot1D(fnum, y_FFTscale, xfnyq, pltTitle, pltXlabel, pltYlabel)

    # // *-----------------------------------------------------------------* //

    # #########################################################################
    # // *---Pulse Wave  Plot - source signal array vs. FFT MAG out array---*
    # #########################################################################

    # // *-----------------------------------------------------------------* //


# // *---------------------------------------------------------------------* //
# // *---------------------------------------------------------------------* //


plt.show()




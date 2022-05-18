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

# /////////////////////////////////////////////////////////////////////////////
# #############################################################################
# begin : Generate source waveforms
# #############################################################################
# \\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\


# // *---------------------------------------------------------------------* //
# generate simple mono sin waves
if genMonoSin == 1:

    print('\n::Mono Sine waves::')
    print('generated mono sin signals @ 2.5K and 5K Hz')
    
    plotMonoSin = 1

    xl = 1.56   # set signal length
    monoSinFreq = 560.0

    # Test mono sin generator function
    
    # use global signal length
    monoSinOut = np.array([y for y in tbWavGen.monosin(monoSinFreq)])
    # use specific signal length (** must include sample rate before length)
    monoSinOutxl = np.array([y for y in tbWavGen.monosin(monoSinFreq, sr, xl)])
    
    if createWavFile == 1:
        
        wavGenMonoSinOut = wavGenOutDir + 'MonoSinOut.wav'
        write_wav(wavGenMonoSinOut, monoSinOut, sr)
        
        wavGenMonoSinOutxl = wavGenOutDir + 'MonoSinOutxl.wav'
        write_wav(wavGenMonoSinOutxl, monoSinOutxl, sr)    

    # Test mono sin Array function
    
    monoSinArrayOut = tbWavGen.monosinArray(monoSinFreq)
    monoSinArrayOutxl = tbWavGen.monosinArray(monoSinFreq, sr, xl)
    
    if createWavFile == 1:
        
        wavGenMonoSinOut = wavGenOutDir + 'MonoSinArrayOut.wav'
        write_wav(wavGenMonoSinOut, monoSinArrayOut, sr)
        
        wavGenMonoSinOutxl = wavGenOutDir + 'MonoSinArrayOutxl.wav'
        write_wav(wavGenMonoSinOutxl, monoSinArrayOutxl, sr)
    
else:
    plotMonoSin = 0
    
    
# // *---------------------------------------------------------------------* //    
# generate simple mono tri wave
if genMonoTri == 1:
    
    print('\n::Mono Tri waves::')
    
    plotMonoTri = 1    

    xl = 0.23
    monoTriFreq = 700.0

    # Test mono sin generator function
    
    # use global signal length
    monoTriOut = np.array([y for y in tbWavGen.monotri(monoTriFreq, sr, xl)])
    
    # use specific signal length
    monoTriOutxl = np.array([y for y in tbWavGen.monotri(monoTriFreq, sr, xl)])
    
    # pdb.set_trace()
    
    if createWavFile == 1:
        
        wavGenMonoTriOut = wavGenOutDir+'MonoTriOut.wav'
        write_wav(wavGenMonoTriOut, monoTriOut, sr)
        
        wavGenMonoTriOutxl = wavGenOutDir+'MonoTriOutxl.wav'
        write_wav(wavGenMonoTriOutxl, monoTriOutxl, sr)    

    # Test mono sin Array function
    
    monoTriArrayOut = tbWavGen.monotriArray(monoTriFreq, sr, xl)
    monoTriArrayOutxl = tbWavGen.monotriArray(monoTriFreq, sr, xl)
    
    if createWavFile == 1:
        
        wavGenMonoTriOut = wavGenOutDir+'MonoTriArrayOut.wav'
        write_wav(wavGenMonoTriOut, monoTriArrayOut, sr)
        
        wavGenMonoTriOutxl = wavGenOutDir+'MonoTriArrayOutxl.wav'
        write_wav(wavGenMonoTriOutxl, monoTriArrayOutxl, sr)

    
else:
    plotMonoTri = 0
    

# // *---------------------------------------------------------------------* //
# generate LFO signals
if genLFO == 1:
    
    # Create new object with lower sample...
    
    plotLFO = 1
    
    xlLFO = 13
    srLfo = 500.0
    TLfo = 1.0 / srLfo
    tbLFOGen = wavGen.xodWavGen(numSamples, srLfo)
    
    lfoFreqArray1 = [0.125, 0.5]

    # LFO (sin) generator function
    
    # use specific signal length
    LFO_L = np.array([y for y in tbLFOGen.monosin(lfoFreqArray1[0], srLfo, xlLFO)])
    LFO_R = np.array([y for y in tbLFOGen.monosin(lfoFreqArray1[1], srLfo, xlLFO)])
    testLFOdual = [LFO_L, LFO_R]    
    
    
else:
    plotLFO = 0


# // *---------------------------------------------------------------------* //
# generate array of sin waves
if genSinArray == 1:
    
    plotSinArray = 1
    
    odmkTestFreqArray2_1 = [444.0, 1776.0]
    odmkTestFreqArray2_2 = [2500.0, 5000.0]    
    
    odmkTestFreqArray5_1 = [1000.0, 3000.0, 5000.0, 7000.0, 9000.0]
    odmkTestFreqArray5_2 = [666.0, 777.7, 2300.0, 6000.0, 15600.0]
    odmkTestFreqArray5_3 = [3200.0, 6400.0, 9600.0, 12800.0, 16000.0]
    
    # orthogonal sets
    odmkTestFreqArray3_1 = [3200.0, 6400.0, 9600.0]
    odmkTestFreqArray7_1 = [3200.0, 6400.0, 9600.0, 12800.0, 16000.0, 19200.0,  22400.0]

    print('\n::Multi Sine source::')
    # testFreqs = [666.0, 777.7, 2300.0, 6000.0, 15600.0]
    testFreqs = odmkTestFreqArray5_2
    numFreqSinArray = len(testFreqs)

    print('Frequency Array (Hz):')
    print(testFreqs)

    sinArray = np.array([])
    for freq in testFreqs:
        sinArray = np.concatenate((sinArray, tbWavGen.monosinArray(freq)))
    sinArray = sinArray.reshape((numFreqSinArray, numSamples))

    print('generated array of sin signals "sinArray"')
    

else:
    plotSinArray = 0

    
# // *---------------------------------------------------------------------* //    
# generate array of sin waves
if genOrthoSinArray == 1:
    
    plotOrthoSinArray = 1    
    
    
#    numOrtFreqs = 7
#    nCzn = cyclicZn(numOrtFreqs)
#    
#    
#    nOrthogonalArray = np.array([])
#    for c in range(numOrtFreqs):
#        nCznPh = np.arctan2(nCzn[c].imag, nCzn[c].real)
#        nOrthogonalArray = np.append(nOrthogonalArray, (fs*nCznPh)/(2*np.pi))
    
    # Example orthogonal array:
    # >>> nCzn =7
    # array([[ 1.00000000+0.j        ],
    #       [ 0.62348980+0.78183148j],
    #       [-0.22252093+0.97492791j],
    #       [-0.90096887+0.43388374j],
    #       [-0.90096887-0.43388374j],
    #       [-0.22252093-0.97492791j],
    #       [ 0.62348980-0.78183148j]])

    # generate a set of orthogonal frequencies

    print('\n::Orthogonal Multi Sine source::')

    # for n freqs, use 2n+1 => skip DC and negative freqs!
    # ex. for cyclicZn(15), we want to use czn[1, 2, 3, ... 7]

    numOrthoFreq = 7
    czn = cyclicZn(2*numOrthoFreq + 1)

    orthoFreqArray = np.array([])
    for c in range(1, numOrthoFreq+1):
        cznph = np.arctan2(czn[c].imag, czn[c].real)
        cznFreq = (sr*cznph)/(2*np.pi)
        orthoFreqArray = np.append(orthoFreqArray, cznFreq)

    print('Orthogonal Frequency Array (Hz):')
    print(orthoFreqArray)
    
    # pdb.set_trace()

    orthoSinArray = np.array([])
    for freq in orthoFreqArray:
        orthoSinArray = np.concatenate((orthoSinArray, tbWavGen.monosinArray(freq)))
    orthoSinArray = orthoSinArray.reshape((numOrthoFreq, numSamples))

    print('generated array of orthogonal sin signals "orthoSinArray"')

else:
    plotOrthoSinArray = 0

# // *---------------------------------------------------------------------* //
# generate a composite signal of an array of sin waves "sum of sines"
if genCompositeSinArray == 1:
    
    plotCompositeSinArray = 1

    print('\n::Composite Multi Sine source::')

    # user:

    odmkTestFreqArray2_1 = [444.0, 1776.0]
    odmkTestFreqArray2_2 = [2500.0, 5000.0]    
    
    odmkTestFreqArray5_1 = [1000.0, 2000.0, 3000.0, 4000.0, 5000.0]
    odmkTestFreqArray5_2 = [666.0, 777.7, 2300.0, 6000.0, 15600.0]
    odmkTestFreqArray5_3 = [3200.0, 6400.0, 9600.0, 12800.0, 16000.0]
    
    # orthogonal sets
    odmkTestFreqArray3_1 = [3200.0, 6400.0, 9600.0]
    odmkTestFreqArray7_1 = [3200.0, 6400.0, 9600.0, 12800.0, 16000.0, 19200.0,  22400.0]

    freqArray = odmkTestFreqArray5_1

    multiSinOut = tbWavGen.multiSinArray(freqArray)

    print('Generated composite sine signal: sinOrth5Comp1')
    print('generated a Composite array of sin signals "orthoSinComp1"')
    
    if createWavFile == 1:
        wavGenMultiSinOut = wavGenOutDir+'multiSinOut.wav'
        write_wav(wavGenMultiSinOut, multiSinOut, sr)

else:
    plotCompositeSinArray = 0


# // *---------------------------------------------------------------------* //

# #############################################################################
# Wavetable Synthesis
# #############################################################################

# // *---------------------------------------------------------------------* //
# generate output from wavetable oscillator
if genWavetableOsc == 1:
    
    plotWavetableOsc = 1

    print('\n::Wavetable oscillator::')

    # defined above
    # shape:
    # 1=sin,     2=cos,     3=tri,     4=saw-up,    5=saw-dn,
    # 6=exp-up,  7=exp-dn,  8=log-up,  9=log-dn,    10=cheby,
    # 11=pulse1, 12=pulse2, 13=pulse3, 14=pulse4,   15=user

    shape = 'sin'
    
    # testFreq1 = 777.0
    # testPhase1 = 0

    freqCtrl = testFreq1
    phaseCtrl = testPhase1
    
    # run wavetable osc function with default (no) quantization
    [xodOsc, xodOsc90, xodSqrPulse] = tbWavGen.xodWTOsc(numSamples, shape, freqCtrl, phaseCtrl)

else:
    plotWavetableOsc = 0


# // *---------------------------------------------------------------------* //

# #############################################################################
# begin : Pulse Width Modulations
# #############################################################################

# // *---------------------------------------------------------------------* //
# generate pulse width modulated square wav
if genPWMOsc == 1:
    
    plotPWMOsc = 1

    print('\n::Pulse Width Modulated square wav::')

    ''' generates pulse width modulated square wav
        usage:
            >>phaseInc = output base freq - Fo / Fs
            >>pulseWidth = % of cycle [0 - 100] '''

    pwmOutFreq = 1000

    pwmCtrlFreq = 100

    phaseInc = pwmOutFreq/sr

    pulseWidthCtrl = 0.5 * tbWavGen.monosin(pwmCtrlFreq) + 0.5

    pwmOut = np.zeros([numSamples])

    for i in range(numSamples):
        pwmOut[i] = tbWavGen.pulseWidthMod(phaseInc, pulseWidthCtrl[i])

else:
    plotPWMOsc = 0

# /////////////////////////////////////////////////////////////////////////////
# #############################################################################
# begin : Plotting
# #############################################################################
# \\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\

print('\n')
print('// *--------------------------------------------------------------* //')
print('// *---::Plotting::---*')
print('// *--------------------------------------------------------------* //')


# // *---------------------------------------------------------------------* //

if plotMonoSin == 1:
    
    # // *---------------------------------------------------------------------* //
    # // *---Mono Sin plots---*
    
    # FFT length
    N = 4096

    y = monoSinOut[0:N]
    y_FFT = sp.fft(y)
    # scale and format FFT out for plotting
    y_FFTscale = 2.0/N * np.abs(y_FFT[0:int(N/2)])
    # y_Mag = np.abs(y_FFT)
    # y_Phase = np.arctan2(y_FFT.imag, y_FFT.real)
    
    yArr = monoSinArrayOut[0:N]
    yArr_FFT = sp.fft(yArr)
    # scale and format FFT out for plotting
    yArr_FFTscale = 2.0/N * np.abs(yArr_FFT[0:int(N/2)])

    y_combined = np.concatenate((y, yArr))
    y_combined = y_combined.reshape((2, N))
    
    yfft_combined = np.concatenate((y_FFTscale, yArr_FFTscale))
    yfft_combined = yfft_combined.reshape((2, int(N/2)))

    # define a sub-range for wave plot visibility
    # tLen = 500
    
    # Plot y1 time domain:
    fnum = 1
    pltTitle = 'plt1: Mono Sin '+str(monoSinFreq)+' Hz, (first '+str(N)+' samples)'
    pltXlabel = 'time'
    pltYlabel = 'Magnitude'
    
    # define a linear space from 0 to 1/2 Fs for x-axis:
    xaxis = np.linspace(0, N, N)
    xodplt.xodPlot1D(fnum, y, xaxis, pltTitle, pltXlabel, pltYlabel)
    
    # plot y freq domain:
    fnum = 2
    pltTitle = 'plt2: FFT Mag Mono Sin '+str(monoSinFreq)+' Hz'
    pltXlabel = 'Frequency: 0 - '+str(sr / 2)+' Hz'
    pltYlabel = 'Magnitude (scaled by 2/N)'
    
    # define a linear space from 0 to 1/2 Fs for x-axis:
    xfnyq = np.linspace(0.0, 1.0/(2.0*T), int(N/2))
    xodplt.xodPlot1D(fnum, y_FFTscale, xfnyq, pltTitle, pltXlabel, pltYlabel)

    # Plot y1 time domain:
    fnum = 3
    pltTitle = 'plt3: Mono Sin (generator vs. array) '+str(monoSinFreq)+' Hz, (first '+str(N)+' samples)'
    pltXlabel = 'time'
    pltYlabel = 'Magnitude'
    
    # define a linear space from 0 to 1/2 Fs for x-axis:
    xaxis = np.linspace(0, N, N)
    xodplt.xodMultiPlot1D(fnum, y_combined, xaxis, pltTitle, pltXlabel, pltYlabel, colorMap='gnuplot')
    
    # plot y freq domain:
    fnum = 4
    pltTitle = 'plt4: FFT Mag Mono Sin '+str(monoSinFreq)+' Hz'
    pltXlabel = 'Frequency: 0 - '+str(sr / 2)+' Hz'
    pltYlabel = 'Magnitude (scaled by 2/N)'

    # define a linear space from 0 to 1/2 Fs for x-axis:
    xfnyq = np.linspace(0.0, 1.0/(2.0*T), int(N/2))
    xodplt.xodMultiPlot1D(fnum, yfft_combined, xfnyq, pltTitle, pltXlabel, pltYlabel, colorMap='gnuplot')
    
    
# // *---------------------------------------------------------------------* //
    
if plotMonoTri == 1:

    # // *---------------------------------------------------------------------* //   
    # // *---Mono Triangle wave plots---*

    # FFT length
    N = 4096

    yTri = monoTriOut[0:N]
    yTri_FFT = sp.fft(yTri)
    # scale and format FFT out for plotting
    yTri_FFTscale = 2.0/N * np.abs(yTri_FFT[0:int(N/2)])
    # y_Mag = np.abs(y_FFT)
    # y_Phase = np.arctan2(y_FFT.imag, y_FFT.real)
    
    yTriArr = monoTriArrayOut[0:N]
    yTriArr_FFT = sp.fft(yTriArr)
    # scale and format FFT out for plotting
    yTriArr_FFTscale = 2.0/N * np.abs(yTriArr_FFT[0:int(N/2)])

    yTri_combined = np.concatenate((yTri, yTriArr))
    yTri_combined = yTri_combined.reshape((2, N))
    
    yTrifft_combined = np.concatenate((yTri_FFTscale, yTriArr_FFTscale))
    yTrifft_combined = yTrifft_combined.reshape((2, int(N/2)))

    # define a sub-range for wave plot visibility
    # tLen = 500
    
    # Plot y1 time domain:
    fnum = 5
    pltTitle = 'plt1: Mono Tri '+str(monoTriFreq)+' Hz, (first '+str(N)+' samples)'
    pltXlabel = 'time'
    pltYlabel = 'Magnitude'
    
    # define a linear space from 0 to 1/2 Fs for x-axis:
    xaxis = np.linspace(0, N, N)
    xodplt.xodPlot1D(fnum, yTri, xaxis, pltTitle, pltXlabel, pltYlabel)
    
    # plot y freq domain:
    fnum = 6
    pltTitle = 'plt2: FFT Mag Mono Tri '+str(monoTriFreq)+' Hz'
    pltXlabel = 'Frequency: 0 - '+str(sr / 2)+' Hz'
    pltYlabel = 'Magnitude (scaled by 2/N)'
    
    # define a linear space from 0 to 1/2 Fs for x-axis:
    xfnyq = np.linspace(0.0, 1.0/(2.0*T), int(N/2))
    xodplt.xodPlot1D(fnum, yTri_FFTscale, xfnyq, pltTitle, pltXlabel, pltYlabel)

    # Plot y1 time domain:
    fnum = 7
    pltTitle = 'plt3: Mono Tri (generator vs. array) '+str(monoTriFreq)+' Hz, (first '+str(N)+' samples)'
    pltXlabel = 'time'
    pltYlabel = 'Magnitude'
    
    # define a linear space from 0 to 1/2 Fs for x-axis:
    xaxis = np.linspace(0, N, N)
    xodplt.xodMultiPlot1D(fnum, yTri_combined, xaxis, pltTitle, pltXlabel, pltYlabel, colorMap='gnuplot')
    
    # plot y freq domain:
    fnum = 8
    pltTitle = 'plt4: FFT Mag Mono Tri '+str(monoTriFreq)+' Hz'
    pltXlabel = 'Frequency: 0 - '+str(sr / 2)+' Hz'
    pltYlabel = 'Magnitude (scaled by 2/N)'

    # define a linear space from 0 to 1/2 Fs for x-axis:
    xfnyq = np.linspace(0.0, 1.0/(2.0*T), int(N/2))
    xodplt.xodMultiPlot1D(fnum, yTrifft_combined, xfnyq, pltTitle, pltXlabel, pltYlabel, colorMap='gnuplot')    

    
# // *---------------------------------------------------------------------* //    
    
if plotLFO == 1:
    
    # // *---------------------------------------------------------------------* //   
    # // *---Mono Triangle wave plots---*

    # Test FFT length
    N = 4096
    
    # check if signal length is less than Test FFT length
    if N > len(LFO_L):
        print("WARNING: LFO_L length is less than test FFT length -> zero padding LFO_L")
        testLFO_L = np.append(LFO_L, np.zeros(N - len(LFO_L)))
    else:
        testLFO_L = LFO_L[0:N]
    # define a linear space from 0 to 1/2 Fs for x-axis:
    xaxis = np.linspace(0, N, N)

    fnum = 21
    pltTitle = 'SigGen output: LFO_L: N Hz sine LFO waveform (first '+str(N)+' samples)'
    pltXlabel = 'time'
    pltYlabel = 'Magnitude (scaled)'

    # pdb.set_trace()
    
    xodplt.xodPlot1D(fnum, testLFO_L, xaxis, pltTitle, pltXlabel, pltYlabel)    

    # // *-----------------------------------------------------------------* //

    LFO_L_FFT = sp.fft(testLFO_L)
    # scale and format FFT out for plotting
    LFO_L_FFTscale = 2.0/N * np.abs(LFO_L_FFT[0:int(N/2)])

    fnum = 22
    pltTitle = 'FFT Mag: LFO_L_FFTscale N Hz sine LFO'
    pltXlabel = 'Frequency: 0 - '+str(srLfo / 2)+' Hz'
    pltYlabel = 'Magnitude (scaled by 2/N)'
    
    # define a linear space from 0 to 1/2 Fs for x-axis:
    xfnyq = np.linspace(0.0, 1.0 / (2.0 * TLfo), N/2)
    
    xodplt.xodPlot1D(fnum, LFO_L_FFTscale, xfnyq, pltTitle, pltXlabel, pltYlabel)    


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


odmkPwmPlots = 0
if odmkPwmPlots == 1:

    # // *---------------------------------------------------------------------* //
    # // *---Mono FFT plots---*
    # // *---------------------------------------------------------------------* //

    # define a sub-range for wave plot visibility
    tLen = 48000
    
    fnum = 1
    pltTitle = 'Input Signal pwmOut (first '+str(tLen)+' samples)'
    pltXlabel = 'pwm1_FFTscale: '+str(pwmOutFreq)+' Hz'
    pltYlabel = 'Magnitude'

    sig = pwmOut[0:tLen]
    # define a linear space from 0 to 1/2 Fs for x-axis:
    xaxis = np.linspace(0, tLen, tLen)

    xodplt.xodPlot1D(fnum, sig, xaxis, pltTitle, pltXlabel, pltYlabel)

    pwm1 = pwmOut[0:N]
    pwm1_FFT = sp.fft(pwm1)
    # scale and format FFT out for plotting
    pwm1_FFTscale = 2.0/N * np.abs(pwm1_FFT[0:int(N/2)])

    fnum = 60
    pltTitle = 'FFT Mag: pwm1_FFTscale '+str(pwmOutFreq)+' Hz'
    pltXlabel = 'Frequency: 0 - '+str(sr / 2)+' Hz'
    pltYlabel = 'Magnitude (scaled by 2/N)'
    
    # sig <= direct
    
    # define a linear space from 0 to 1/2 Fs for x-axis:
    xfnyq = np.linspace(0.0, 1.0/(2.0*T), N/2)
    
    xodplt.xodPlot1D(fnum, pwm1_FFTscale, xfnyq, pltTitle, pltXlabel, pltYlabel)    

# // *---------------------------------------------------------------------* //
# // *---------------------------------------------------------------------* //


plt.show()




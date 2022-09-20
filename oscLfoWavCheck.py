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

# sys.path.insert(1, rootDir+'/xodma')
# from xodmaAudioTools import write_wav


# temp python debugger - use >>>pdb.set_trace() to set break
import pdb

# run this command to de-embed plots
# %matplotlib qt


# \\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\
# // *---------------------------------------------------------------------* //
# // *-- Functions--*
# // *---------------------------------------------------------------------* //

def cyclicZn(n):
    """ calculates the Zn roots of unity """
    cZn = np.zeros((n, 1))*(0+0j)    # column vector of zero complex values
    for k in range(n):
        # z(k) = e^(((k)*2*pi*1j)/n)        # Define cyclic group Zn points
        cZn[k] = np.cos((k * 2 * np.pi) / n) + np.sin((k * 2 * np.pi) / n) * 1j   # Euler's identity

    return cZn


def arrayFromFile(fname):
    """ reads .dat(.txt) data into Numpy array
        fname is the path+name to existing text file
        example: newArray = arrayFromFile('/eschei/data/mydata_in.dat') """

    datalist = []
    with open(fname, mode='r') as infile:
        for line in infile.readlines():
            datalist.append(float(line))
    arrayNm = np.array(datalist)

    fileSrc = os.path.split(fname)[1]
    filePath = os.path.split(fname)[0]

    print('\nLoaded file: ' + fileSrc)
    print('\nfile path: ' + filePath)

    flength = len(list(arrayNm))    # replace with better method??
    print('Length of data = ' + str(flength))

    return arrayNm

# /////////////////////////////////////////////////////////////////////////////
# #############################################################################
# begin Test:
# #############################################################################
# \\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\


# // *---------------------------------------------------------------------* //
# // *---::Set Parameters for test output::---*')

# length of x in seconds:
xLength = 5  # 0.0024
# audio sample rate:
sr = 48000
# Fclk = 100000000    # FPGA DDS test

# sample period
T = 1.0 / sr
# Tfclk = 1.0 / Fclk

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

# FIXIT FIXIT - redundant (xodWavGen_tb) replace with verifications
# select generated waveforms
importWavSource = 0
genMonoSin = 0
genMonoTri = 0
genLFO = 0
genSinArray = 0
genCompositeSinArray = 0


# select input waveform verification
checkOrthogSinArray = 1


genWavetableOsc = 0
if genWavetableOsc == 1:
    # shape:
    # 1=sin,     2=cos,     3=tri,     4=saw-up,    5=saw-dn,
    # 6=exp-up,  7=exp-dn,  8=log-up,  9=log-dn,    10=cheby,
    # 11=pulse1, 12=pulse2, 13=pulse3, 14=pulse4,   15=user
    shape = 'pulse3'

genPWMOsc = 0


# // *---------------------------------------------------------------------* //
# // *---::Instantiate clock & signal Generator objects::---*

tbClocks = clks.XodClocks(xLength, sr, bpm, framesPerSec)

numSamples = tbClocks.totalSamples
tbclkDownBeats = tbClocks.clkDownBeats()

wavGenOutDir = audioOutDir + 'wavGenOutDir/'

tbWavGen = wavGen.xodWavGen(sr, xLength, wavGenOutDir)

# // *---------------------------------------------------------------------* //
# // *---------------------------------------------------------------------* //
# Load Source Waveform from file and Plot
if importWavSource == 1:

    srcWav1 = '/home/eschei/xodmk/xodCode/xodHLS/audio/data/output/dds_lfoSin_out.txt'

    arrayWav1 = arrayFromFile(srcWav1)
    plotWavSource = 1

else:
    plotWavSource = 0

# // *---------------------------------------------------------------------* //
# // *---:: Generate Model waveforms ::---*

# generate simple mono sin waves
if genMonoSin == 1:

    print('\n::Mono Sine waves::')
    print('generated mono sin signals @ 2.5K and 5K Hz')

    plotMonoSin = 1

    xl = 1.56  # set signal length
    monoSinFreq = 560.0

    # Test mono sin generator function

    # use global signal length
    monoSinOut = np.array([y for y in tbWavGen.monosin(monoSinFreq)])
    # use specific signal length (** must include sample rate before length)
    monoSinOutxl = np.array([y for y in tbWavGen.monosin(monoSinFreq, sr, xl)])

    # Test mono sin Array function

    monoSinArrayOut = tbWavGen.monosinArray(monoSinFreq)
    monoSinArrayOutxl = tbWavGen.monosinArray(monoSinFreq, sr, xl)


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

    monoTriArrayOut = tbWavGen.monotriArray(monoTriFreq, sr, xl)
    monoTriArrayOutxl = tbWavGen.monotriArray(monoTriFreq, sr, xl)


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
    odmkTestFreqArray7_1 = [3200.0, 6400.0, 9600.0, 12800.0, 16000.0, 19200.0, 22400.0]

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
    odmkTestFreqArray7_1 = [3200.0, 6400.0, 9600.0, 12800.0, 16000.0, 19200.0, 22400.0]

    freqArray = odmkTestFreqArray5_1

    multiSinOut = tbWavGen.multiSinArray(freqArray)

    print('Generated composite sine signal: sinOrth5Comp1')
    print('generated a Composite array of sin signals "orthoSinComp1"')


else:
    plotCompositeSinArray = 0


# // *---------------------------------------------------------------------* //

# /////////////////////////////////////////////////////////////////////////////
# begin : input waveform verification
# \\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\

# // *---------------------------------------------------------------------* //
# generate orthogonal array of sin waves - 4 CHannels
if checkOrthogSinArray == 1:

    plotCheckOrthogSinArray4CH = 1

    numOrthogChannels = 4

    print('\n::Check Orthogonal Multi Sine source 4CH::')

    srcWav = []
    srcArrayTmp = np.array([])
    srcWavArray = np.array([])

    srcWav.append('/home/eschei/xodmk/xodCode/xodHLS/audio/data/output/dds_sin_out_T1_0.txt')
    srcWav.append('/home/eschei/xodmk/xodCode/xodHLS/audio/data/output/dds_sin_out_T1_1.txt')
    srcWav.append('/home/eschei/xodmk/xodCode/xodHLS/audio/data/output/dds_sin_out_T1_2.txt')
    srcWav.append('/home/eschei/xodmk/xodCode/xodHLS/audio/data/output/dds_sin_out_T1_3.txt')

    for c in range(numOrthogChannels):
        srcArrayTmp = arrayFromFile(srcWav[c])
        srcWavArray = np.concatenate((srcWavArray, srcArrayTmp))
    srcWavArray = srcWavArray.reshape(numOrthogChannels, len(srcArrayTmp))

    # for n freqs, use 2n+1 => skip DC and negative freqs!
    # ex. for cyclicZn(15), we want to use czn[1, 2, 3, ... 7]

    czn = cyclicZn(2 * numOrthogChannels + 1)

    orthogFreqArray = np.array([])
    for c in range(1, numOrthogChannels + 1):
        cznph = np.arctan2(czn[c].imag, czn[c].real)
        cznFreq = (sr/2 * cznph) / (2 * np.pi)  # limit max freq to 12 KHz (HLS step freq ctrl uses +/- 24bits)
        cznFreqInt = int(cznFreq)
        orthogFreqArray = np.append(orthogFreqArray, cznFreqInt)

    print('Orthogonal Frequency Array 4CH (Hz):')
    print(orthogFreqArray)

    # pdb.set_trace()

    orthogSinArray = np.array([])
    for freq in orthogFreqArray:
        orthogSinArray = np.concatenate((orthogSinArray, tbWavGen.monosinArray(freq)))
    orthogSinArray = orthogSinArray.reshape((numOrthogChannels, numSamples))

    print('generated 4CH array of orthogonal sin signals "orthogSinArray4CH"')

else:
    plotCheckOrthogSinArray = 0


# // *---------------------------------------------------------------------* //


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
# // *--- Input Source Waveform Plots---*

if plotWavSource == 1:
    arrayWav1

    # FFT length
    N = 4096

    y = arrayWav1[0:N]
    y_FFT = np.fft.fft(y)
    # scale and format FFT out for plotting
    y_FFTscale = 2.0 / N * np.abs(y_FFT[0:int(N / 2)])
    # y_Mag = np.abs(y_FFT)
    # y_Phase = np.arctan2(y_FFT.imag, y_FFT.real)

    # define a sub-range for wave plot visibility
    # tLen = 500

    # Plot y1 time domain:
    fnum = 1
    pltTitle = 'plt1: Input Wave Source (first ' + str(N) + ' samples)'
    pltXlabel = 'time'
    pltYlabel = 'Magnitude'

    # define a linear space from 0 to 1/2 Fs for x-axis:
    xaxis = np.linspace(0, N, N)
    xodplt.xodPlot1D(fnum, y, xaxis, pltTitle, pltXlabel, pltYlabel)

    # plot y freq domain:
    fnum = 2
    pltTitle = 'plt2: FFT Mag Input Wave Source'
    pltXlabel = 'Frequency: 0 - ' + str(sr / 2) + ' Hz'
    pltYlabel = 'Magnitude (scaled by 2/N)'

    # define a linear space from 0 to 1/2 Fs for x-axis:
    xfnyq = np.linspace(0.0, 1.0 / (2.0 * T), int(N / 2))
    xodplt.xodPlot1D(fnum, y_FFTscale, xfnyq, pltTitle, pltXlabel, pltYlabel)


# // *---------------------------------------------------------------------* //

if plotMonoSin == 1:
    # // *---------------------------------------------------------------------* //
    # // *---Mono Sin plots---*

    # FFT length
    N = 4096

    y = monoSinOut[0:N]
    y_FFT = np.fft.fft(y)
    # scale and format FFT out for plotting
    y_FFTscale = 2.0 / N * np.abs(y_FFT[0:int(N / 2)])
    # y_Mag = np.abs(y_FFT)
    # y_Phase = np.arctan2(y_FFT.imag, y_FFT.real)

    yArr = monoSinArrayOut[0:N]
    yArr_FFT = np.fft.fft(yArr)
    # scale and format FFT out for plotting
    yArr_FFTscale = 2.0 / N * np.abs(yArr_FFT[0:int(N / 2)])

    y_combined = np.concatenate((y, yArr))
    y_combined = y_combined.reshape((2, N))

    yfft_combined = np.concatenate((y_FFTscale, yArr_FFTscale))
    yfft_combined = yfft_combined.reshape((2, int(N / 2)))

    # define a sub-range for wave plot visibility
    # tLen = 500

    # Plot y1 time domain:
    fnum = 10
    pltTitle = 'plt1: Mono Sin ' + str(monoSinFreq) + ' Hz, (first ' + str(N) + ' samples)'
    pltXlabel = 'time'
    pltYlabel = 'Magnitude'

    # define a linear space from 0 to 1/2 Fs for x-axis:
    xaxis = np.linspace(0, N, N)
    xodplt.xodPlot1D(fnum, y, xaxis, pltTitle, pltXlabel, pltYlabel)

    # plot y freq domain:
    fnum = 11
    pltTitle = 'plt2: FFT Mag Mono Sin ' + str(monoSinFreq) + ' Hz'
    pltXlabel = 'Frequency: 0 - ' + str(sr / 2) + ' Hz'
    pltYlabel = 'Magnitude (scaled by 2/N)'

    # define a linear space from 0 to 1/2 Fs for x-axis:
    xfnyq = np.linspace(0.0, 1.0 / (2.0 * T), int(N / 2))
    xodplt.xodPlot1D(fnum, y_FFTscale, xfnyq, pltTitle, pltXlabel, pltYlabel)

    # Plot y1 time domain:
    fnum = 12
    pltTitle = 'plt3: Mono Sin (generator vs. array) ' + str(monoSinFreq) + ' Hz, (first ' + str(N) + ' samples)'
    pltXlabel = 'time'
    pltYlabel = 'Magnitude'

    # define a linear space from 0 to 1/2 Fs for x-axis:
    xaxis = np.linspace(0, N, N)
    xodplt.xodMultiPlot1D(fnum, y_combined, xaxis, pltTitle, pltXlabel, pltYlabel, colorMap='gnuplot')

    # plot y freq domain:
    fnum = 13
    pltTitle = 'plt4: FFT Mag Mono Sin ' + str(monoSinFreq) + ' Hz'
    pltXlabel = 'Frequency: 0 - ' + str(sr / 2) + ' Hz'
    pltYlabel = 'Magnitude (scaled by 2/N)'

    # define a linear space from 0 to 1/2 Fs for x-axis:
    xfnyq = np.linspace(0.0, 1.0 / (2.0 * T), int(N / 2))
    xodplt.xodMultiPlot1D(fnum, yfft_combined, xfnyq, pltTitle, pltXlabel, pltYlabel, colorMap='gnuplot')

# // *---------------------------------------------------------------------* //

if plotMonoTri == 1:
    # // *---------------------------------------------------------------------* //
    # // *---Mono Triangle wave plots---*

    # FFT length
    N = 4096

    yTri = monoTriOut[0:N]
    yTri_FFT = np.fft.fft(yTri)
    # scale and format FFT out for plotting
    yTri_FFTscale = 2.0 / N * np.abs(yTri_FFT[0:int(N / 2)])
    # y_Mag = np.abs(y_FFT)
    # y_Phase = np.arctan2(y_FFT.imag, y_FFT.real)

    yTriArr = monoTriArrayOut[0:N]
    yTriArr_FFT = np.fft.fft(yTriArr)
    # scale and format FFT out for plotting
    yTriArr_FFTscale = 2.0 / N * np.abs(yTriArr_FFT[0:int(N / 2)])

    yTri_combined = np.concatenate((yTri, yTriArr))
    yTri_combined = yTri_combined.reshape((2, N))

    yTrifft_combined = np.concatenate((yTri_FFTscale, yTriArr_FFTscale))
    yTrifft_combined = yTrifft_combined.reshape((2, int(N / 2)))

    # define a sub-range for wave plot visibility
    # tLen = 500

    # Plot y1 time domain:
    fnum = 20
    pltTitle = 'plt1: Mono Tri ' + str(monoTriFreq) + ' Hz, (first ' + str(N) + ' samples)'
    pltXlabel = 'time'
    pltYlabel = 'Magnitude'

    # define a linear space from 0 to 1/2 Fs for x-axis:
    xaxis = np.linspace(0, N, N)
    xodplt.xodPlot1D(fnum, yTri, xaxis, pltTitle, pltXlabel, pltYlabel)

    # plot y freq domain:
    fnum = 21
    pltTitle = 'plt2: FFT Mag Mono Tri ' + str(monoTriFreq) + ' Hz'
    pltXlabel = 'Frequency: 0 - ' + str(sr / 2) + ' Hz'
    pltYlabel = 'Magnitude (scaled by 2/N)'

    # define a linear space from 0 to 1/2 Fs for x-axis:
    xfnyq = np.linspace(0.0, 1.0 / (2.0 * T), int(N / 2))
    xodplt.xodPlot1D(fnum, yTri_FFTscale, xfnyq, pltTitle, pltXlabel, pltYlabel)

    # Plot y1 time domain:
    fnum = 22
    pltTitle = 'plt3: Mono Tri (generator vs. array) ' + str(monoTriFreq) + ' Hz, (first ' + str(N) + ' samples)'
    pltXlabel = 'time'
    pltYlabel = 'Magnitude'

    # define a linear space from 0 to 1/2 Fs for x-axis:
    xaxis = np.linspace(0, N, N)
    xodplt.xodMultiPlot1D(fnum, yTri_combined, xaxis, pltTitle, pltXlabel, pltYlabel, colorMap='gnuplot')

    # plot y freq domain:
    fnum = 23
    pltTitle = 'plt4: FFT Mag Mono Tri ' + str(monoTriFreq) + ' Hz'
    pltXlabel = 'Frequency: 0 - ' + str(sr / 2) + ' Hz'
    pltYlabel = 'Magnitude (scaled by 2/N)'

    # define a linear space from 0 to 1/2 Fs for x-axis:
    xfnyq = np.linspace(0.0, 1.0 / (2.0 * T), int(N / 2))
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

    fnum = 30
    pltTitle = 'SigGen output: LFO_L: N Hz sine LFO waveform (first ' + str(N) + ' samples)'
    pltXlabel = 'time'
    pltYlabel = 'Magnitude (scaled)'

    # pdb.set_trace()

    xodplt.xodPlot1D(fnum, testLFO_L, xaxis, pltTitle, pltXlabel, pltYlabel)

    # // *-----------------------------------------------------------------* //

    LFO_L_FFT = np.fft.fft(testLFO_L)
    # scale and format FFT out for plotting
    LFO_L_FFTscale = 2.0 / N * np.abs(LFO_L_FFT[0:int(N / 2)])

    fnum = 31
    pltTitle = 'FFT Mag: LFO_L_FFTscale N Hz sine LFO'
    pltXlabel = 'Frequency: 0 - ' + str(srLfo / 2) + ' Hz'
    pltYlabel = 'Magnitude (scaled by 2/N)'

    # define a linear space from 0 to 1/2 Fs for x-axis:
    xfnyq = np.linspace(0.0, 1.0 / (2.0 * TLfo), N / 2)

    xodplt.xodPlot1D(fnum, LFO_L_FFTscale, xfnyq, pltTitle, pltXlabel, pltYlabel)

# // *---------------------------------------------------------------------* //

if plotSinArray == 1:

    # // *---------------------------------------------------------------------* //
    # // *---Array of Sines wave plots---*

    # Test FFT length
    N = 4096

    tLen = N

    numFreqs = numFreqSinArray  # defined above for gen of sinArray

    yArray = np.array([])
    yScaleArray = np.array([])
    # for h in range(len(sinArray[0, :])):
    for h in range(numFreqs):
        yFFT = np.fft.fft(sinArray[h, 0:N])
        yArray = np.concatenate((yArray, yFFT))
        yScaleArray = np.concatenate((yScaleArray, 2.0 / N * np.abs(yFFT[0:int(N / 2)])))
    yArray = yArray.reshape((numFreqs, N))
    yScaleArray = yScaleArray.reshape((numFreqs, int(N / 2)))

    fnum = 40
    pltTitle = 'Input Signals: sinArray (first ' + str(tLen) + ' samples)'
    pltXlabel = 'sinArray time-domain wav'
    pltYlabel = 'Magnitude'

    # define a linear space from 0 to 1/2 Fs for x-axis:
    xaxis = np.linspace(0, tLen, tLen)

    # pdb.set_trace()

    xodplt.xodMultiPlot1D(fnum, sinArray, xaxis, pltTitle, pltXlabel, pltYlabel, colorMap='hsv')

    fnum = 41
    pltTitle = 'FFT Mag: yScaleArray multi-osc '
    pltXlabel = 'Frequency: 0 - ' + str(sr / 2) + ' Hz'
    pltYlabel = 'Magnitude (scaled by 2/N)'

    # define a linear space from 0 to 1/2 Fs for x-axis:
    xfnyq = np.linspace(0.0, 1.0 / (2.0 * T), int(N / 2))

    xodplt.xodMultiPlot1D(fnum, yScaleArray, xfnyq, pltTitle, pltXlabel, pltYlabel, colorMap='hsv')

# // *---------------------------------------------------------------------* //

if plotCheckOrthogSinArray4CH == 1:

    # // *---------------------------------------------------------------------* //
    # // *---Array of Orthogonal Sines wave plots---*

    # Test FFT length
    N = 4096

    assert len(orthogSinArray[0, :]) >= N
    assert len(srcWavArray[0, :]) >= N

    tLen = N

    numFreqs = numOrthogChannels

    # orthogSinArray - python generated waveforms
    # srcWavArray - imported waveforms from c++ file i/o

    yOrthogArray = np.array([])
    yOrthogScaleArray = np.array([])
    ySrcWavArray = np.array([])
    ySrcWavScaleArray = np.array([])

    # for h in range(len(sinArray[0, :])):
    for h in range(numFreqs):
        yOrthogFFT = np.fft.fft(orthogSinArray[h, 0:N])
        yOrthogArray = np.concatenate((yOrthogArray, yOrthogFFT))
        yOrthogScaleArray = np.concatenate((yOrthogScaleArray, 2.0 / N * np.abs(yOrthogFFT[0:int(N / 2)])))
        ySrcWavFFT = np.fft.fft(srcWavArray[h, 0:N])
        ySrcWavArray = np.concatenate((ySrcWavArray, ySrcWavFFT))
        ySrcWavScaleArray = np.concatenate((ySrcWavScaleArray, 2.0 / N * np.abs(ySrcWavFFT[0:int(N / 2)])))
    yOrthogArray = yOrthogArray.reshape((numFreqs, N))
    yOrthogScaleArray = yOrthogScaleArray.reshape(numFreqs, (int(N / 2)))
    ySrcWavArray = ySrcWavArray.reshape((numFreqs, N))
    ySrcWavScaleArray = ySrcWavScaleArray.reshape(numFreqs, (int(N / 2)))

    # // *---------------------------------------------------------------------* //
    # *** plot python generated waveforms ***
    fnum = 50
    pltTitle = 'Input Signals: orthoSinArray (first ' + str(tLen) + ' samples)'
    pltXlabel = 'orthoSinArray time-domain wav'
    pltYlabel = 'Magnitude'

    # define a linear space from 0 to 1/2 Fs for x-axis:
    xaxis = np.linspace(0, tLen, tLen)

    xodplt.xodMultiPlot1D(fnum, orthogSinArray, xaxis, pltTitle, pltXlabel, pltYlabel, colorMap='hsv')

    fnum = 51
    pltTitle = 'FFT Mag: yOrthogScaleArray multi-osc '
    pltXlabel = 'Frequency: 0 - ' + str(sr / 2) + ' Hz'
    pltYlabel = 'Magnitude (scaled by 2/N)'

    # define a linear space from 0 to 1/2 Fs for x-axis:
    xfnyq = np.linspace(0.0, 1.0 / (2.0 * T), int(N / 2))

    xodplt.xodMultiPlot1D(fnum, yOrthogScaleArray, xfnyq, pltTitle, pltXlabel, pltYlabel, colorMap='hsv')

    # // *---------------------------------------------------------------------* //
    # *** plot imported imported waveforms ***
    fnum = 52
    pltTitle = 'Input Signals: srcWavArray (first ' + str(tLen) + ' samples)'
    pltXlabel = 'srcWavArray time-domain wav'
    pltYlabel = 'Magnitude'

    # define a linear space from 0 to 1/2 Fs for x-axis:
    xaxis = np.linspace(0, tLen, tLen)

    xodplt.xodMultiPlot1D(fnum, srcWavArray, xaxis, pltTitle, pltXlabel, pltYlabel, colorMap='hsv')

    fnum = 53
    pltTitle = 'FFT Mag: ySrcWavScaleArray multi-osc '
    pltXlabel = 'Frequency: 0 - ' + str(sr / 2) + ' Hz'
    pltYlabel = 'Magnitude (scaled by 2/N)'

    # define a linear space from 0 to 1/2 Fs for x-axis:
    xfnyq = np.linspace(0.0, 1.0 / (2.0 * T), int(N / 2))

    xodplt.xodMultiPlot1D(fnum, ySrcWavScaleArray, xfnyq, pltTitle, pltXlabel, pltYlabel, colorMap='hsv')


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

    fnum = 60
    pltTitle = 'SigGen output: sinOrth5Comp1 Composite waveform (first ' + str(tLen) + ' samples)'
    pltXlabel = 'time'
    pltYlabel = 'Magnitude (scaled)'

    xodplt.xodPlot1D(fnum, sig, xaxis, pltTitle, pltXlabel, pltYlabel)

    # // *-----------------------------------------------------------------* //

    ySinComp1 = multiSinOut[0:N]

    sinComp1_FFT = np.fft.fft(ySinComp1)
    sinComp1_FFTscale = 2.0 / N * np.abs(sinComp1_FFT[0:int(N / 2)])

    fnum = 61
    pltTitle = 'FFT Mag: sinComp1_FFTscale Composite waveform'
    pltXlabel = 'Frequency: 0 - ' + str(sr / 2) + ' Hz'
    pltYlabel = 'Magnitude (scaled by 2/N)'

    # sig <= direct

    # define a linear space from 0 to 1/2 Fs for x-axis:
    xfnyq = np.linspace(0.0, 1.0 / (2.0 * T), int(N / 2))

    xodplt.xodPlot1D(fnum, sinComp1_FFTscale, xfnyq, pltTitle, pltXlabel, pltYlabel)

    # // *-----------------------------------------------------------------* //


# // *---------------------------------------------------------------------* //
# // *---------------------------------------------------------------------* //


plt.show()




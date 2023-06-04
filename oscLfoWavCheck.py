# -*- coding: utf-8 -*-
# *****************************************************************************
# /////////////////////////////////////////////////////////////////////////////
# \\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\
# *****************************************************************************
#
# __::(( oscLfoWavCheck.py ))::__
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


# /////////////////////////////////////////////////////////////////////////////
# // *---------------------------------------------------------------------* //
# // *--- Functions ---*
# // *---------------------------------------------------------------------* //
# \\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\

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
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# begin Test:
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# \\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\

# // *---------------------------------------------------------------------* //
# // *---:: Set Parameters for test output ::---*')

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

fnum = 0

# /////////////////////////////////////////////////////////////////////////////
# xodWavGen waveform verification Select / CTRL
# \\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\

# select generated waveforms
genMonoSin = 0
genMonoTri = 0
genLFO = 0
genSinArray = 0
genCompositeSinArray = 0

genWavetableOsc = 0
if genWavetableOsc == 1:
    # shape:
    # 1=sin,     2=cos,     3=tri,     4=saw-up,    5=saw-dn,
    # 6=exp-up,  7=exp-dn,  8=log-up,  9=log-dn,    10=cheby,
    # 11=pulse1, 12=pulse2, 13=pulse3, 14=pulse4,   15=user
    shape = 'pulse3'

genPWMOsc = 0


# /////////////////////////////////////////////////////////////////////////////
# Input waveform verification Select / CTRL
# \\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\

# * Read waveform data from file (.txt / .dat file sample array)
# * plot time & freq domain

# Test Basic: Read single channel waveform data - plot time & freq domain
importWavSource = 0

# Example: HLS 4 Channel Orthogonal sin wav output vs. Python reference
checkOrthogSinArray4CH = 1
checkDDSXNSinArray4CH = 1

# Test X0: Single channel LFO
checkLFO1CH = 0

# Test-1: Test-1: 4 Channel LFO Array
checkLFO4CH = 0

# Test-2: 4 Channel Orthogonal Freq LFO Array
checkOrthogFreqLFO4CH = 0


# /////////////////////////////////////////////////////////////////////////////
# Instantiate clock & signal Generator objects
# \\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\

tbClocks = clks.XodClocks(xLength, sr, bpm, framesPerSec)

numSamples = tbClocks.totalSamples              # used for self-generating wavform tests
tbclkDownBeats = tbClocks.clkDownBeats()

wavGenOutDir = audioOutDir + 'wavGenOutDir/'

tbWavGen = wavGen.xodWavGen(sr, xLength, wavGenOutDir)


# /////////////////////////////////////////////////////////////////////////////
# xodWavGen waveform verification Signal Processing
# \\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\

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
# // *---:: generate simple mono tri wave ::---*

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
# // *---:: generate LFO signals ::---*

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
# // *---:: generate array of sin waves ::---*

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
# // *---:: generate composite signal of array of sin waves "sum of sines" ::---*

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


# /////////////////////////////////////////////////////////////////////////////
# Input waveform verification signal processing
# \\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\

# // *---------------------------------------------------------------------* //
# // *---:: Load Source Waveform from file and Plot ::---*

if importWavSource == 1:

    # Direct from xodHLS data/output results:
    srcWav1 = '/home/eschei/xodmk/xodCode/xodHLS/audio/data/output/dds_lfoSin_out.txt'

    # Saved xodHLS data/output results for archive consistency
    # srcWav1 = '/home/eschei/xodmk/xodCode/xodPython/data/res/xodHLS_resOut_ver/hlsBasicSingleChannel/dds_lfoSin_out.txt'

    arrayWav1 = arrayFromFile(srcWav1)
    plotWavSource = 1

else:
    plotWavSource = 0

# // *---------------------------------------------------------------------* //
# // *---:: Check Orthogonal array of sin waves - 4 CHannels ::---*

if checkOrthogSinArray4CH == 1:

    print('\n::Check Orthogonal Multi Sine source 4CH::')

    plotCheckOrthogSinArray4CH = 1
    numOrthogChannels = 4

    srcWav = []
    srcArrayTmp = np.array([])
    srcOrthogSinArray = np.array([])

    # frequency array = {2666, 5333, 7999, 10666};
    # Direct from xodHLS data/output results:
    srcWav.append('/home/eschei/xodmk/xodCode/xodHLS/audio/data/output/xodDDSXN_res/dds_sin_out_T1_0.txt')
    srcWav.append('/home/eschei/xodmk/xodCode/xodHLS/audio/data/output/xodDDSXN_res/dds_sin_out_T1_1.txt')
    srcWav.append('/home/eschei/xodmk/xodCode/xodHLS/audio/data/output/xodDDSXN_res/dds_sin_out_T1_2.txt')
    srcWav.append('/home/eschei/xodmk/xodCode/xodHLS/audio/data/output/xodDDSXN_res/dds_sin_out_T1_3.txt')

    for c in range(numOrthogChannels):
        srcArrayTmp = arrayFromFile(srcWav[c])
        if c == 0:
            srcLength = len(srcArrayTmp)
            print(f'\nLength of Source Signal (CH0) = {str(srcLength)}')
        else:
            assert len(srcArrayTmp) == srcLength, \
                f'Length of srcWav doesn\'t match testLength, srcWavLength: {len(srcArrayTmp)}'
        srcOrthogSinArray = np.concatenate((srcOrthogSinArray, srcArrayTmp))
    srcOrthogSinArray = srcOrthogSinArray.reshape(numOrthogChannels, len(srcArrayTmp))

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

    srcTime = srcLength / sr

    refOrthogSinArray = np.array([])
    for freq in orthogFreqArray:
        refOrthogSinArray = np.concatenate((refOrthogSinArray, tbWavGen.monosinArray(freq, sr, srcTime)))
    refOrthogSinArray = refOrthogSinArray.reshape((numOrthogChannels, srcLength))

    print('generated 4CH array of orthogonal sin signals "orthogSinArray4CH"')

    # Outputs Arrays:
    # refOrthogSinArray, srcOrthogSinArray

else:
    plotCheckOrthogSinArray4CH = 0


# // *---------------------------------------------------------------------* //
# // *---:: Check Orthogonal array of sin waves - 4 CHannels ::---*

if checkDDSXNSinArray4CH == 1:

    print('\n::Check Orthogonal Multi Sine source 4CH::')

    plotCheckDDSXNSinArray4CH = 1
    numChannels = 4

    srcWav = []
    srcArrayTmp = np.array([])
    srcDDSXNSinArray = np.array([])

    # frequency array = {10000, 10666, 10666, 11332};
    # Direct from xodHLS data/output results:
    srcWav.append('/home/eschei/xodmk/xodCode/xodHLS/audio/data/output/xodDDSXN_res/dds_sin_out_T2_0.txt')
    srcWav.append('/home/eschei/xodmk/xodCode/xodHLS/audio/data/output/xodDDSXN_res/dds_sin_out_T2_1.txt')
    srcWav.append('/home/eschei/xodmk/xodCode/xodHLS/audio/data/output/xodDDSXN_res/dds_sin_out_T2_2.txt')
    srcWav.append('/home/eschei/xodmk/xodCode/xodHLS/audio/data/output/xodDDSXN_res/dds_sin_out_T2_3.txt')

    for c in range(numChannels):
        srcArrayTmp = arrayFromFile(srcWav[c])
        if c == 0:
            srcLength = len(srcArrayTmp)
            print(f'\nLength of Source Signal (CH0) = {str(srcLength)}')
        else:
            assert len(srcArrayTmp) == srcLength, \
                f'Length of srcWav doesn\'t match testLength, srcWavLength: {len(srcArrayTmp)}'
        srcDDSXNSinArray = np.concatenate((srcDDSXNSinArray, srcArrayTmp))
    srcDDSXNSinArray = srcDDSXNSinArray.reshape(numChannels, len(srcArrayTmp))

    DDSXNFreqArray = np.array([10000, 10666, 10666, 11332])

    print('DDSXN Frequency Array 4CH (Hz):')
    print(DDSXNFreqArray)

    # pdb.set_trace()

    srcTime = srcLength / sr

    refDDSXNSinArray = np.array([])
    for freq in DDSXNFreqArray:
        refDDSXNSinArray = np.concatenate((refDDSXNSinArray, tbWavGen.monosinArray(freq, sr, srcTime)))
    refDDSXNSinArray = refDDSXNSinArray.reshape((numChannels, srcLength))

    print('generated 4CH array of DDSXN sin signals "DDSXNSinArray4CH"')

    # Outputs Arrays:
    # refDDSXNSinArray, srcDDSXNSinArray

else:
    plotCheckDDSXNSinArray4CH = 0


# // *---------------------------------------------------------------------* //
# // *---:: CTest X0: Single channel LFO ::---*

if checkLFO1CH == 1:

    print('\n::Check Single channel LFO::')

    plotCheckLFO1CH = 1

    lfoWavData1CH = np.array([])
    srcLfoArray = np.array([])

    if 1:
        # Direct from xodHLS data/output results:
        lfoWav1CH = '/home/eschei/xodmk/xodCode/xodHLS/audio/data/output/xodLFOXN_res/lfo_out_T0.txt'
        # lfoWav1CH = '/home/eschei/xodmk/xodCode/xodHLS/audio/data/output/xodLFOXN_res/lfo_out_T1_0.txt'

    else:
        # Saved xodHLS data/output results for archive consistency
        lfoWav1CH = '/home/eschei/xodmk/xodCode/xodPython/data/res/xodHLS_resOut_ver/hlsLfoArray4CH_res/lfo_out_T1_0.txt'

    srcLfoData1CH = arrayFromFile(lfoWav1CH)
    lfoLength = len(srcLfoData1CH)
    print(f'\nLength of Source Signal (CH0) = {str(lfoLength)}')

    # ** Assumes frequency array matches with imported frequency array **
    lfoRefFreq = 560
    print(f'\nLFO Frequency = {str(lfoRefFreq)}')

    lfoTime = lfoLength / sr

    refLfoData1CH = tbWavGen.monosinArray(lfoRefFreq, sr, lfoTime)

    print('generated 1CH LFO signal "srcLfoData1CH"')

    # Outputs Arrays:
    # refLfoData1CH, srcLfoData1CH


else:
    plotCheckLFO1CH = 0


# // *---------------------------------------------------------------------* //
# // *---:: Test-1: 4 Channel LFO Array ::---*
# // *---:: [111.11111111 222.22222222 333.33333333 444.44444444] ::---*

if checkLFO4CH == 1:

    print('\n::4 Channel LFO Array::')

    plotCheckLFO4CH = 1
    numLfoChannels = 4

    lfoWav = []
    lfoArrayTmp = np.array([])
    srcLfoArray = np.array([])

    if 1:
        # Direct from xodHLS data/output results:
        lfoWav.append('/home/eschei/xodmk/xodCode/xodHLS/audio/data/output/lfo_out_T1_0.txt')
        lfoWav.append('/home/eschei/xodmk/xodCode/xodHLS/audio/data/output/lfo_out_T1_1.txt')
        lfoWav.append('/home/eschei/xodmk/xodCode/xodHLS/audio/data/output/lfo_out_T1_2.txt')
        lfoWav.append('/home/eschei/xodmk/xodCode/xodHLS/audio/data/output/lfo_out_T1_3.txt')
    else:
        # Saved xodHLS data/output results for archive consistency
        lfoWav.append('/home/eschei/xodmk/xodCode/xodPython/data/res/xodHLS_resOut_ver/hlsLfoArray4CH_res/lfo_out_T1_0.txt')
        lfoWav.append('/home/eschei/xodmk/xodCode/xodPython/data/res/xodHLS_resOut_ver/hlsLfoArray4CH_res/lfo_out_T1_1.txt')
        lfoWav.append('/home/eschei/xodmk/xodCode/xodPython/data/res/xodHLS_resOut_ver/hlsLfoArray4CH_res/lfo_out_T1_2.txt')
        lfoWav.append('/home/eschei/xodmk/xodCode/xodPython/data/res/xodHLS_resOut_ver/hlsLfoArray4CH_res/lfo_out_T1_3.txt')

    for c in range(numLfoChannels):
        lfoArrayTmp = arrayFromFile(lfoWav[c])
        if c == 0:
            lfoLength = len(lfoArrayTmp)
            print(f'\nLength of Source Signal (CH0) = {str(lfoLength)}')
        else:
            assert len(lfoArrayTmp) == lfoLength, \
                f'Length of lfoWav doesn\'t match testLength, srcWavLength: {len(lfoArrayTmp)}'
        srcLfoArray = np.concatenate((srcLfoArray, lfoArrayTmp))
    srcLfoArray = srcLfoArray.reshape(numLfoChannels, len(lfoArrayTmp))

    # ** Assumes frequency array matches with imported frequency array **
    lfoRefFreqArray = np.array([111.11111111, 222.22222222, 333.33333333, 444.44444444])
    print('LFO Frequency Array 4CH (Hz):')
    print(lfoRefFreqArray)

    # pdb.set_trace()

    lfoTime = lfoLength / sr

    refLfoArray = np.array([])
    for freq in lfoRefFreqArray:
        refLfoArray = np.concatenate((refLfoArray, tbWavGen.monosinArray(freq, sr, lfoTime)))
    refLfoArray = refLfoArray.reshape((numLfoChannels, lfoLength))

    print('generated 4CH array of LFO signals "srcLfoArray"')

    # Outputs Arrays:
    # refLfoArray, srcLfoArray

else:
    plotCheckLFO4CH = 0


# // *---------------------------------------------------------------------* //
# // *---:: Test-2: 4 Channel Orthogonal Freq LFO Array ::---*

if checkOrthogFreqLFO4CH == 1:

    plotCheckOrthogFreqLFO4CH = 1
    numLfoChannels = 4
    print('\n::Check Orthogonal Multi Sine source 4CH::')

    orthogFreqlfoWav = []
    orthogFreqlfoArrayTmp = np.array([])
    orthogFreqlfoWavArray = np.array([])

    if 1:
        # Direct from xodHLS data/output results:
        orthogFreqlfoWav.append('/home/eschei/xodmk/xodCode/xodHLS/audio/data/output/lfo_out_T1_0.txt')
        orthogFreqlfoWav.append('/home/eschei/xodmk/xodCode/xodHLS/audio/data/output/lfo_out_T1_1.txt')
        orthogFreqlfoWav.append('/home/eschei/xodmk/xodCode/xodHLS/audio/data/output/lfo_out_T1_2.txt')
        orthogFreqlfoWav.append('/home/eschei/xodmk/xodCode/xodHLS/audio/data/output/lfo_out_T1_3.txt')
    else:
        # Saved xodHLS data/output results for archive consistency
        orthogFreqlfoWav.append('/home/eschei/xodmk/xodCode/xodPython/data/res/xodHLS_resOut_ver/hlsLfoArray4CH_res/lfo_out_T1_0.txt')
        orthogFreqlfoWav.append('/home/eschei/xodmk/xodCode/xodPython/data/res/xodHLS_resOut_ver/hlsLfoArray4CH_res/lfo_out_T1_1.txt')
        orthogFreqlfoWav.append('/home/eschei/xodmk/xodCode/xodPython/data/res/xodHLS_resOut_ver/hlsLfoArray4CH_res/lfo_out_T1_2.txt')
        orthogFreqlfoWav.append('/home/eschei/xodmk/xodCode/xodPython/data/res/xodHLS_resOut_ver/hlsLfoArray4CH_res/lfo_out_T1_3.txt')

    for c in range(numLfoChannels):
        orthogFreqlfoArrayTmp = arrayFromFile(orthogFreqlfoWav[c])
        orthogFreqlfoWavArray = np.concatenate((orthogFreqlfoWavArray, orthogFreqlfoArrayTmp))
    orthogFreqlfoWavArray = orthogFreqlfoWavArray.reshape(numLfoChannels, len(lfoArrayTmp))

    # for n freqs, use 2n+1 => skip DC and negative freqs!
    # ex. for cyclicZn(15), we want to use czn[1, 2, 3, ... 7]

    czn = cyclicZn(2 * numLfoChannels + 1)

    orthogFreqlfoRefFreqArray = np.array([])
    for c in range(1, numLfoChannels + 1):
        cznph = np.arctan2(czn[c].imag, czn[c].real)
        cznFreq = (sr/2 * cznph) / (2 * np.pi)  # limit max freq to 12 KHz (HLS step freq ctrl uses +/- 24bits)
        cznFreqInt = int(cznFreq)
        orthogFreqlfoRefFreqArray = np.append(orthogFreqlfoRefFreqArray, cznFreqInt)

    print('Orthogonal Frequency Array 4CH (Hz):')
    print(orthogFreqlfoRefFreqArray)

    # pdb.set_trace()

    orthogFreqlfoRefArray = np.array([])
    for freq in orthogFreqlfoRefFreqArray:
        orthogFreqlfoRefArray = np.concatenate((orthogFreqlfoRefArray, tbWavGen.monosinArray(freq)))
    orthogFreqlfoRefArray = orthogFreqlfoRefArray.reshape((numLfoChannels, numSamples))

    print('generated 4CH array of orthogonal sin signals "orthogSinArray4CH"')

else:
    plotCheckOrthogFreqLFO4CH = 0

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


# /////////////////////////////////////////////////////////////////////////////
# xodWavGen waveform verification Plotting
# \\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\

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
    fnum = fnum + 1
    pltTitle = 'plt1: Mono Sin ' + str(monoSinFreq) + ' Hz, (first ' + str(N) + ' samples)'
    pltXlabel = 'time'
    pltYlabel = 'Magnitude'

    # define a linear space from 0 to 1/2 Fs for x-axis:
    xaxis = np.linspace(0, N, N)
    xodplt.xodPlot1D(fnum, y, xaxis, pltTitle, pltXlabel, pltYlabel)

    # plot y freq domain:
    fnum = fnum + 1
    pltTitle = 'plt2: FFT Mag Mono Sin ' + str(monoSinFreq) + ' Hz'
    pltXlabel = 'Frequency: 0 - ' + str(sr / 2) + ' Hz'
    pltYlabel = 'Magnitude (scaled by 2/N)'

    # define a linear space from 0 to 1/2 Fs for x-axis:
    xfnyq = np.linspace(0.0, 1.0 / (2.0 * T), int(N / 2))
    xodplt.xodPlot1D(fnum, y_FFTscale, xfnyq, pltTitle, pltXlabel, pltYlabel)

    # Plot y1 time domain:
    fnum = fnum + 1
    pltTitle = 'plt3: Mono Sin (generator vs. array) ' + str(monoSinFreq) + ' Hz, (first ' + str(N) + ' samples)'
    pltXlabel = 'time'
    pltYlabel = 'Magnitude'

    # define a linear space from 0 to 1/2 Fs for x-axis:
    xaxis = np.linspace(0, N, N)
    xodplt.xodMultiPlot1D(fnum, y_combined, xaxis, pltTitle, pltXlabel, pltYlabel, colorMap='gnuplot')

    # plot y freq domain:
    fnum = fnum + 1
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
    fnum = fnum + 1
    pltTitle = 'plt1: Mono Tri ' + str(monoTriFreq) + ' Hz, (first ' + str(N) + ' samples)'
    pltXlabel = 'time'
    pltYlabel = 'Magnitude'

    # define a linear space from 0 to 1/2 Fs for x-axis:
    xaxis = np.linspace(0, N, N)
    xodplt.xodPlot1D(fnum, yTri, xaxis, pltTitle, pltXlabel, pltYlabel)

    # plot y freq domain:
    fnum = fnum + 1
    pltTitle = 'plt2: FFT Mag Mono Tri ' + str(monoTriFreq) + ' Hz'
    pltXlabel = 'Frequency: 0 - ' + str(sr / 2) + ' Hz'
    pltYlabel = 'Magnitude (scaled by 2/N)'

    # define a linear space from 0 to 1/2 Fs for x-axis:
    xfnyq = np.linspace(0.0, 1.0 / (2.0 * T), int(N / 2))
    xodplt.xodPlot1D(fnum, yTri_FFTscale, xfnyq, pltTitle, pltXlabel, pltYlabel)

    # Plot y1 time domain:
    fnum = fnum + 1
    pltTitle = 'plt3: Mono Tri (generator vs. array) ' + str(monoTriFreq) + ' Hz, (first ' + str(N) + ' samples)'
    pltXlabel = 'time'
    pltYlabel = 'Magnitude'

    # define a linear space from 0 to 1/2 Fs for x-axis:
    xaxis = np.linspace(0, N, N)
    xodplt.xodMultiPlot1D(fnum, yTri_combined, xaxis, pltTitle, pltXlabel, pltYlabel, colorMap='gnuplot')

    # plot y freq domain:
    fnum = fnum + 1
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

    fnum = fnum + 1
    pltTitle = 'SigGen output: LFO_L: N Hz sine LFO waveform (first ' + str(N) + ' samples)'
    pltXlabel = 'time'
    pltYlabel = 'Magnitude (scaled)'

    # pdb.set_trace()

    xodplt.xodPlot1D(fnum, testLFO_L, xaxis, pltTitle, pltXlabel, pltYlabel)

    # // *-----------------------------------------------------------------* //

    LFO_L_FFT = np.fft.fft(testLFO_L)
    # scale and format FFT out for plotting
    LFO_L_FFTscale = 2.0 / N * np.abs(LFO_L_FFT[0:int(N / 2)])

    fnum = fnum + 1
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

    fnum = fnum + 1
    pltTitle = 'Input Signals: sinArray (first ' + str(tLen) + ' samples)'
    pltXlabel = 'sinArray time-domain wav'
    pltYlabel = 'Magnitude'

    # define a linear space from 0 to 1/2 Fs for x-axis:
    xaxis = np.linspace(0, tLen, tLen)

    # pdb.set_trace()

    xodplt.xodMultiPlot1D(fnum, sinArray, xaxis, pltTitle, pltXlabel, pltYlabel, colorMap='hsv')

    fnum = fnum + 1
    pltTitle = 'FFT Mag: yScaleArray multi-osc '
    pltXlabel = 'Frequency: 0 - ' + str(sr / 2) + ' Hz'
    pltYlabel = 'Magnitude (scaled by 2/N)'

    # define a linear space from 0 to 1/2 Fs for x-axis:
    xfnyq = np.linspace(0.0, 1.0 / (2.0 * T), int(N / 2))

    xodplt.xodMultiPlot1D(fnum, yScaleArray, xfnyq, pltTitle, pltXlabel, pltYlabel, colorMap='hsv')


# /////////////////////////////////////////////////////////////////////////////
# Input waveform verification Plots
# \\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\

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
    fnum = fnum + 1
    pltTitle = 'plt1: Input Wave Source (first ' + str(N) + ' samples)'
    pltXlabel = 'time'
    pltYlabel = 'Magnitude'

    # define a linear space from 0 to 1/2 Fs for x-axis:
    xaxis = np.linspace(0, N, N)
    xodplt.xodPlot1D(fnum, y, xaxis, pltTitle, pltXlabel, pltYlabel)

    # plot y freq domain:
    fnum = fnum + 1
    pltTitle = 'plt2: FFT Mag Input Wave Source'
    pltXlabel = 'Frequency: 0 - ' + str(sr / 2) + ' Hz'
    pltYlabel = 'Magnitude (scaled by 2/N)'

    # define a linear space from 0 to 1/2 Fs for x-axis:
    xfnyq = np.linspace(0.0, 1.0 / (2.0 * T), int(N / 2))
    xodplt.xodPlot1D(fnum, y_FFTscale, xfnyq, pltTitle, pltXlabel, pltYlabel)


# // *---------------------------------------------------------------------* //
if plotCheckOrthogSinArray4CH == 1:

    # // *---------------------------------------------------------------------* //
    # // *---Array of Orthogonal Sines wave plots---*

    # python reference source array: refOrthogSinArray
    # External input source array: srcOrthogSinArray

    # Test FFT length
    N = 4096

    assert len(refOrthogSinArray[0, :]) >= N,\
        f"Length of ref signal less than FFT length, reflength = {len(refOrthogSinArray[0, :])}"
    assert len(srcOrthogSinArray[0, :]) >= N,\
        f"Length of src signal less than FFT length, srclength = {len(srcOrthogSinArray[0, :])}"

    tLen = N
    numFreqs = numOrthogChannels

    yOrthogArray = np.array([])
    yOrthogScaleArray = np.array([])
    ySrcWavArray = np.array([])
    ySrcWavScaleArray = np.array([])

    # for h in range(len(sinArray[0, :])):
    for h in range(numFreqs):
        # python ref
        yOrthogFFT = np.fft.fft(refOrthogSinArray[h, 0:N])
        yOrthogArray = np.concatenate((yOrthogArray, yOrthogFFT))
        yOrthogScaleArray = np.concatenate((yOrthogScaleArray, 2.0 / N * np.abs(yOrthogFFT[0:int(N / 2)])))
        # imported src
        ySrcWavFFT = np.fft.fft(srcOrthogSinArray[h, 0:N])
        ySrcWavArray = np.concatenate((ySrcWavArray, ySrcWavFFT))
        ySrcWavScaleArray = np.concatenate((ySrcWavScaleArray, 2.0 / N * np.abs(ySrcWavFFT[0:int(N / 2)])))
    # python ref freq domain
    yOrthogArray = yOrthogArray.reshape((numFreqs, N))
    yOrthogScaleArray = yOrthogScaleArray.reshape(numFreqs, (int(N / 2)))
    # imported src freq domain
    ySrcWavArray = ySrcWavArray.reshape((numFreqs, N))
    ySrcWavScaleArray = ySrcWavScaleArray.reshape(numFreqs, (int(N / 2)))

    # // *---------------------------------------------------------------------* //
    # *** plot python generated waveforms ***
    fnum = fnum + 1
    pltTitle = 'Input Signals: refOrthogSinArray (first ' + str(tLen) + ' samples)'
    pltXlabel = 'orthoSinArray time-domain wav'
    pltYlabel = 'Magnitude'

    # define a linear space from 0 to 1/2 Fs for x-axis:
    xaxis = np.linspace(0, tLen, tLen)

    xodplt.xodMultiPlot1D(fnum, refOrthogSinArray, xaxis, pltTitle, pltXlabel, pltYlabel, colorMap='hsv')

    fnum = fnum + 1
    pltTitle = 'FFT Mag: yOrthogScaleArray multi-osc '
    pltXlabel = 'Frequency: 0 - ' + str(sr / 2) + ' Hz'
    pltYlabel = 'Magnitude (scaled by 2/N)'

    # define a linear space from 0 to 1/2 Fs for x-axis:
    xfnyq = np.linspace(0.0, 1.0 / (2.0 * T), int(N / 2))

    xodplt.xodMultiPlot1D(fnum, yOrthogScaleArray, xfnyq, pltTitle, pltXlabel, pltYlabel, colorMap='hsv')

    # // *---------------------------------------------------------------------* //
    # *** plot imported imported waveforms ***
    fnum = fnum + 1
    pltTitle = 'Input Signals: srcOrthogSinArray (first ' + str(tLen) + ' samples)'
    pltXlabel = 'srcWavArray time-domain wav'
    pltYlabel = 'Magnitude'

    # define a linear space from 0 to 1/2 Fs for x-axis:
    xaxis = np.linspace(0, tLen, tLen)

    xodplt.xodMultiPlot1D(fnum, srcOrthogSinArray, xaxis, pltTitle, pltXlabel, pltYlabel, colorMap='hsv')

    fnum = fnum + 1
    pltTitle = 'FFT Mag: ySrcWavScaleArray multi-osc '
    pltXlabel = 'Frequency: 0 - ' + str(sr / 2) + ' Hz'
    pltYlabel = 'Magnitude (scaled by 2/N)'

    # define a linear space from 0 to 1/2 Fs for x-axis:
    xfnyq = np.linspace(0.0, 1.0 / (2.0 * T), int(N / 2))

    xodplt.xodMultiPlot1D(fnum, ySrcWavScaleArray, xfnyq, pltTitle, pltXlabel, pltYlabel, colorMap='hsv')

# // *---------------------------------------------------------------------* //


# // *---------------------------------------------------------------------* //
if plotCheckDDSXNSinArray4CH == 1:

    # // *---------------------------------------------------------------------* //
    # // *---Array of DDSXN Sines wave plots---*

    # python reference source array: refDDSXNSinArray
    # External input source array: srcDDSXNSinArray

    # Test FFT length
    N = 4096

    assert len(refDDSXNSinArray[0, :]) >= N,\
        f"Length of ref signal less than FFT length, reflength = {len(refDDSXNSinArray[0, :])}"
    assert len(srcDDSXNSinArray[0, :]) >= N,\
        f"Length of src signal less than FFT length, srclength = {len(srcDDSXNSinArray[0, :])}"

    tLen = N
    numFreqs = numChannels

    yRefDDSXNArray = np.array([])
    yRefDDSXNScaleArray = np.array([])
    ySrcDDSXNArray = np.array([])
    ySrcDDSXNScaleArray = np.array([])

    # for h in range(len(sinArray[0, :])):
    for h in range(numFreqs):
        # python ref
        yRefDDSXNFFT = np.fft.fft(refDDSXNSinArray[h, 0:N])
        yRefDDSXNArray = np.concatenate((yRefDDSXNArray, yRefDDSXNFFT))
        yRefDDSXNScaleArray = np.concatenate((yRefDDSXNScaleArray, 2.0 / N * np.abs(yRefDDSXNFFT[0:int(N / 2)])))
        # imported src
        ySrcDDSXNFFT = np.fft.fft(srcDDSXNSinArray[h, 0:N])
        ySrcDDSXNArray = np.concatenate((ySrcDDSXNArray, ySrcDDSXNFFT))
        ySrcDDSXNScaleArray = np.concatenate((ySrcDDSXNScaleArray, 2.0 / N * np.abs(ySrcDDSXNFFT[0:int(N / 2)])))
    # python ref freq domain
    yRefDDSXNArray = yRefDDSXNArray.reshape((numFreqs, N))
    yRefDDSXNScaleArray = yRefDDSXNScaleArray.reshape(numFreqs, (int(N / 2)))
    # imported src freq domain
    ySrcDDSXNArray = ySrcDDSXNArray.reshape((numFreqs, N))
    ySrcDDSXNScaleArray = ySrcDDSXNScaleArray.reshape(numFreqs, (int(N / 2)))

    # // *---------------------------------------------------------------------* //
    # *** plot python generated waveforms ***
    fnum = fnum + 1
    pltTitle = 'Input Signals: refDDSXNSinArray (first ' + str(tLen) + ' samples)'
    pltXlabel = 'orthoSinArray time-domain wav'
    pltYlabel = 'Magnitude'

    # define a linear space from 0 to 1/2 Fs for x-axis:
    xaxis = np.linspace(0, tLen, tLen)

    xodplt.xodMultiPlot1D(fnum, refDDSXNSinArray, xaxis, pltTitle, pltXlabel, pltYlabel, colorMap='hsv')

    fnum = fnum + 1
    pltTitle = 'FFT Mag: yRefDDSXNScaleArray multi-osc '
    pltXlabel = 'Frequency: 0 - ' + str(sr / 2) + ' Hz'
    pltYlabel = 'Magnitude (scaled by 2/N)'

    # define a linear space from 0 to 1/2 Fs for x-axis:
    xfnyq = np.linspace(0.0, 1.0 / (2.0 * T), int(N / 2))

    xodplt.xodMultiPlot1D(fnum, yRefDDSXNScaleArray, xfnyq, pltTitle, pltXlabel, pltYlabel, colorMap='hsv')

    # // *---------------------------------------------------------------------* //
    # *** plot imported imported waveforms ***
    fnum = fnum + 1
    pltTitle = 'Input Signals: srcDDSXNSinArray (first ' + str(tLen) + ' samples)'
    pltXlabel = 'srcWavArray time-domain wav'
    pltYlabel = 'Magnitude'

    # define a linear space from 0 to 1/2 Fs for x-axis:
    xaxis = np.linspace(0, tLen, tLen)

    xodplt.xodMultiPlot1D(fnum, srcDDSXNSinArray, xaxis, pltTitle, pltXlabel, pltYlabel, colorMap='hsv')

    fnum = fnum + 1
    pltTitle = 'FFT Mag: ySrcDDSXNScaleArray multi-osc '
    pltXlabel = 'Frequency: 0 - ' + str(sr / 2) + ' Hz'
    pltYlabel = 'Magnitude (scaled by 2/N)'

    # define a linear space from 0 to 1/2 Fs for x-axis:
    xfnyq = np.linspace(0.0, 1.0 / (2.0 * T), int(N / 2))

    xodplt.xodMultiPlot1D(fnum, ySrcDDSXNScaleArray, xfnyq, pltTitle, pltXlabel, pltYlabel, colorMap='hsv')

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

    fnum = fnum + 1
    pltTitle = 'SigGen output: sinOrth5Comp1 Composite waveform (first ' + str(tLen) + ' samples)'
    pltXlabel = 'time'
    pltYlabel = 'Magnitude (scaled)'

    xodplt.xodPlot1D(fnum, sig, xaxis, pltTitle, pltXlabel, pltYlabel)

    # // *-----------------------------------------------------------------* //

    ySinComp1 = multiSinOut[0:N]

    sinComp1_FFT = np.fft.fft(ySinComp1)
    sinComp1_FFTscale = 2.0 / N * np.abs(sinComp1_FFT[0:int(N / 2)])

    fnum = fnum + 1
    pltTitle = 'FFT Mag: sinComp1_FFTscale Composite waveform'
    pltXlabel = 'Frequency: 0 - ' + str(sr / 2) + ' Hz'
    pltYlabel = 'Magnitude (scaled by 2/N)'

    # sig <= direct

    # define a linear space from 0 to 1/2 Fs for x-axis:
    xfnyq = np.linspace(0.0, 1.0 / (2.0 * T), int(N / 2))

    xodplt.xodPlot1D(fnum, sinComp1_FFTscale, xfnyq, pltTitle, pltXlabel, pltYlabel)

    # // *-----------------------------------------------------------------* //

# // *---------------------------------------------------------------------* //
if plotCheckLFO1CH == 1:

    # // *---------------------------------------------------------------------* //
    # // *--- Test X0: Single channel LFO ---*

    # python reference source array: refLfoData1CH
    # External input source array: srcLfoData1CH

    # Test FFT length
    N = 4096

    assert len(refLfoData1CH) >= N, \
        f"Length of ref signal less than FFT length, reflength = {len(refLfoData1CH)}"
    assert len(srcLfoData1CH) >= N, \
        f"Length of src signal less than FFT length, srclength = {len(srcLfoData1CH)}"

    tLen = N

    # python ref
    yRefLfoFFT = np.fft.fft(refLfoData1CH[0:N])
    yRefLfoFFTScale = 2.0 / N * np.abs(yRefLfoFFT[0:int(N / 2)])
    # imported src
    ySrcLfoFFT = np.fft.fft(srcLfoData1CH[0:N])
    ySrcLfoFFTScale = 2.0 / N * np.abs(ySrcLfoFFT[0:int(N / 2)])

    # // *---------------------------------------------------------------------* //
    # *** plot python generated waveforms ***
    fnum = fnum + 1
    pltTitle = 'Input Signals: refLfoData1CH (first ' + str(tLen) + ' samples)'
    pltXlabel = 'LFO time-domain wav'
    pltYlabel = 'Magnitude'

    # define a linear space from 0 to 1/2 Fs for x-axis:
    xaxis = np.linspace(0, tLen, tLen)

    xodplt.xodPlot1D(fnum, refLfoData1CH, xaxis, pltTitle, pltXlabel, pltYlabel)

    fnum = fnum + 1
    pltTitle = 'FFT Mag: yRefLfoFFTScale '
    pltXlabel = 'Frequency: 0 - ' + str(sr / 2) + ' Hz'
    pltYlabel = 'Magnitude (scaled by 2/N)'

    # define a linear space from 0 to 1/2 Fs for x-axis:
    xfnyq = np.linspace(0.0, 1.0 / (2.0 * T), int(N / 2))

    xodplt.xodPlot1D(fnum, yRefLfoFFTScale, xfnyq, pltTitle, pltXlabel, pltYlabel)

    # // *---------------------------------------------------------------------* //
    # *** plot imported imported waveforms ***
    fnum = fnum + 1
    pltTitle = 'Input Signals: srcLfoData1CH (first ' + str(tLen) + ' samples)'
    pltXlabel = 'srcLfoData1CH time-domain wav'
    pltYlabel = 'Magnitude'

    # define a linear space from 0 to 1/2 Fs for x-axis:
    xaxis = np.linspace(0, tLen, tLen)

    xodplt.xodPlot1D(fnum, srcLfoData1CH, xaxis, pltTitle, pltXlabel, pltYlabel)

    fnum = fnum + 1
    pltTitle = 'FFT Mag: srcLfoData1CH '
    pltXlabel = 'Frequency: 0 - ' + str(sr / 2) + ' Hz'
    pltYlabel = 'Magnitude (scaled by 2/N)'

    # define a linear space from 0 to 1/2 Fs for x-axis:
    xfnyq = np.linspace(0.0, 1.0 / (2.0 * T), int(N / 2))

    xodplt.xodPlot1D(fnum, ySrcLfoFFTScale, xfnyq, pltTitle, pltXlabel, pltYlabel)


# // *---------------------------------------------------------------------* //
if plotCheckLFO4CH == 1:

    # // *---------------------------------------------------------------------* //
    # // *--- Test-1: 4 Channel LFO Array plots ---*

    # python reference source array: refLfoArray
    # External input source array: srcLfoArray

    # Test FFT length
    N = 4096

    assert len(refLfoArray[0, :]) >= N, \
        f"Length of ref signal less than FFT length, reflength = {len(refLfoArray[0, :])}"
    assert len(srcLfoArray[0, :]) >= N, \
        f"Length of src signal less than FFT length, srclength = {len(srcLfoArray[0, :])}"

    tLen = N
    numFreqs = numLfoChannels

    yRefLfoFFTArray = np.array([])
    yRefLfoFFTScaleArray = np.array([])
    ySrcLfoFFTArray = np.array([])
    ySrcLfoFFTScaleArray = np.array([])

    # for h in range(len(sinArray[0, :])):
    for h in range(numFreqs):
        # python ref
        yRefLfoFFT = np.fft.fft(refLfoArray[h, 0:N])
        yRefLfoFFTArray = np.concatenate((yRefLfoFFTArray, yRefLfoFFT))
        yRefLfoFFTScaleArray = np.concatenate((yRefLfoFFTScaleArray, 2.0 / N * np.abs(yRefLfoFFT[0:int(N / 2)])))
        # imported src
        ySrcLfoFFT = np.fft.fft(srcLfoArray[h, 0:N])
        ySrcLfoFFTArray = np.concatenate((ySrcLfoFFTArray, ySrcLfoFFT))
        ySrcLfoFFTScaleArray = np.concatenate((ySrcLfoFFTScaleArray, 2.0 / N * np.abs(ySrcLfoFFT[0:int(N / 2)])))
    # python ref freq domain
    yRefLfoFFTArray = yRefLfoFFTArray.reshape((numFreqs, N))
    yRefLfoFFTScaleArray = yRefLfoFFTScaleArray.reshape(numFreqs, (int(N / 2)))
    # imported src freq domain
    ySrcLfoFFTArray = ySrcLfoFFTArray.reshape((numFreqs, N))
    ySrcLfoFFTScaleArray = ySrcLfoFFTScaleArray.reshape(numFreqs, (int(N / 2)))

    # // *---------------------------------------------------------------------* //
    # *** plot python generated waveforms ***
    fnum = fnum + 1
    pltTitle = 'Input Signals: refOrthogSinArray (first ' + str(tLen) + ' samples)'
    pltXlabel = 'orthoSinArray time-domain wav'
    pltYlabel = 'Magnitude'

    # define a linear space from 0 to 1/2 Fs for x-axis:
    xaxis = np.linspace(0, tLen, tLen)

    xodplt.xodMultiPlot1D(fnum, refLfoArray, xaxis, pltTitle, pltXlabel, pltYlabel, colorMap='hsv')

    fnum = fnum + 1
    pltTitle = 'FFT Mag: yOrthogScaleArray multi-osc '
    pltXlabel = 'Frequency: 0 - ' + str(sr / 2) + ' Hz'
    pltYlabel = 'Magnitude (scaled by 2/N)'

    # define a linear space from 0 to 1/2 Fs for x-axis:
    xfnyq = np.linspace(0.0, 1.0 / (2.0 * T), int(N / 2))

    xodplt.xodMultiPlot1D(fnum, yRefLfoFFTScaleArray, xfnyq, pltTitle, pltXlabel, pltYlabel, colorMap='hsv')

    # // *---------------------------------------------------------------------* //
    # *** plot imported imported waveforms ***
    fnum = fnum + 1
    pltTitle = 'Input Signals: srcOrthogSinArray (first ' + str(tLen) + ' samples)'
    pltXlabel = 'srcWavArray time-domain wav'
    pltYlabel = 'Magnitude'

    # define a linear space from 0 to 1/2 Fs for x-axis:
    xaxis = np.linspace(0, tLen, tLen)

    xodplt.xodMultiPlot1D(fnum, srcLfoArray, xaxis, pltTitle, pltXlabel, pltYlabel, colorMap='hsv')

    fnum = fnum + 1
    pltTitle = 'FFT Mag: ySrcWavScaleArray multi-osc '
    pltXlabel = 'Frequency: 0 - ' + str(sr / 2) + ' Hz'
    pltYlabel = 'Magnitude (scaled by 2/N)'

    # define a linear space from 0 to 1/2 Fs for x-axis:
    xfnyq = np.linspace(0.0, 1.0 / (2.0 * T), int(N / 2))

    xodplt.xodMultiPlot1D(fnum, ySrcLfoFFTScaleArray, xfnyq, pltTitle, pltXlabel, pltYlabel, colorMap='hsv')

# // *---------------------------------------------------------------------* //

if plotCheckOrthogFreqLFO4CH == 1:
    # // *---------------------------------------------------------------------* //
    # // *---Array of Orthogonal Sines wave plots---*
    # python reference source array: refLfoArray
    # External input source array: srcLfoArray

    # Test FFT length
    N = 4096

    # .....

# // *---------------------------------------------------------------------* //


# // *---------------------------------------------------------------------* //
# // *---------------------------------------------------------------------* //


plt.show()




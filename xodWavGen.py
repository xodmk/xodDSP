# -*- coding: utf-8 -*-
# *****************************************************************************
# /////////////////////////////////////////////////////////////////////////////
# \\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\
# *****************************************************************************
#
# __::((xodWavGen.py))::__
#
# Python Signal Generator
#
# Outputs either generator objects, or a numpy arrays
#
# optional .wav output
# optional CSV output
# optonal Spectral Plotting
#
# *****************************************************************************
# /////////////////////////////////////////////////////////////////////////////
# \\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\
# *****************************************************************************

import os
import csv
import numpy as np
#import scipy as sp


# temp python debugger - use >>>pdb.set_trace() to set break
import pdb


# // *---------------------------------------------------------------------* //

# /////////////////////////////////////////////////////////////////////////////
# #############################################################################
# *---::XODMK Waveform Generator 1::---*
# #############################################################################
# \\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\

 

class xodWavGen:
    ''' xodmk Wave Table Generator version 1
        outputs a fs periodic signal normalized to +/- 1
        usage: mySigGen = odmkSigGen1(sigLength, sr, sigType=sin, cmplx=False)
        sigLength => total number of samples
        sr => signal sample rate
        sigType =>
        <<sin, cos, tri, sqr, saw-up, saw-dn, exp-up, exp-dn, log-up, log-dn>>
        cmplx => complex signal - real + imag, 90 deg phase shift
        usage:
        >>import odmkWavGen1 as wavGen
        >>wavGen1OutDir = audioOutDir+'wavGen1OutDir/'
        >>tbWavGen = wavGen.odmkWavGen1(numSamples, fs, wavGen1OutDir)
        >>testFreq1 = 2500.0
        >>testFreq2 = 5000.0
        >>sin5K = tbWavGen.monosin(testFreq2)
        >>cos2_5K = tbWavGen.monocos(testFreq1)
    '''

    def __init__(self, sr=48000, sigLength=1, outDir='None'):

        # *---set primary parameters from inputs---*

        if outDir != 'None':
            self.xodmkWavGenOutDir = outDir
            os.makedirs(outDir, exist_ok=True)
        else:
            self.rootDir = os.path.dirname(os.path.abspath(__file__))
            self.xodmkWavGenOutDir = os.path.join(self.rootDir, "xodmkWavGenOutDir/")
            os.makedirs(self.xodmkWavGenOutDir, exist_ok=True)
        # myfile_path = os.path.join(self.rootDir, "myfile.txt")

        self.sr = sr
        self.sigLength = sigLength
        self.sampleLength = int(np.ceil(sigLength * sr))


        self.phaseAcc = 0



    # // *-----------------------------------------------------------------* //
    # // *---generate sin/cos waveforms
    # // *-----------------------------------------------------------------* //

    def monosin(self, freq, sr='None', xlength='None'):
        ''' generates mono sin wav with frequency = "freq"
            optional fs parameter: default = obj global fs
            usage:
            >>tbSigGen = sigGen.odmkSigGen1(numSamples, fs)
            >>testFreq = 5000.0
            >>sin5K = tbSigGen.monosin(testFreq2) '''

        if sr != 'None':
            sinsr = sr
        else:
            sinsr = self.sr
            
        if xlength != 'None':
            xSamples = int(np.ceil(xlength * sinsr))
        else:
            xSamples = self.sampleLength
            
        #pdb.set_trace()

        # sample period
        T = 1.0 / sinsr

        # generate sin samples
        x = np.linspace(0.0, xSamples*T, xSamples)
        for i in range(xSamples):
            yield np.sin(freq * 2.0*np.pi * x[i])
    
    
    def monosinArray(self, freq, sr='None', xlength='None'):
        ''' outputs array of mono sin wav with frequency = "freq"
            optional fs parameter: default = obj global fs
            usage:
            >>tbSigGen = sigGen.odmkSigGen1(numSamples, fs)
            >>testFreq = 5000.0
            >>sin5K = tbSigGen.monosin(testFreq2) '''

        if sr != 'None':
            sinArrsr = sr
        else:
            sinArrsr = self.sr
            
        if xlength != 'None':
            xSamples = int(np.ceil(xlength * sinArrsr))
        else:
            xSamples = self.sampleLength

        # sample period
        T = 1.0 / sinArrsr

        # create sin array
        x = np.linspace(0.0, xSamples*T, xSamples)
        monosin = np.sin(freq * 2.0*np.pi * x)

        return monosin
    

    def monocos(self, freq, sr='None', xlength='None'):
        ''' generates mono cos wav with frequency = "freq"
            optional sr parameter: default = obj global sr
            usage:
            >>tbSigGen = sigGen.odmkSigGen1(numSamples, sr)
            >>testFreq = 5000.0
            >>sin5K = tbSigGen.monocos(testFreq2) '''

        if sr != 'None':
            cossr = sr
        else:
            cossr = self.sr
            
        if xlength != 'None':
            xSamples = int(np.ceil(xlength * cossr))
        else:
            xSamples = self.sampleLength

        # sample period
        T = 1.0 / cossr

        # create composite sin source
        x = np.linspace(0.0, xSamples*T, xSamples)
        monocos = np.cos(freq * 2.0*np.pi * x)

        return monocos


    def multiSin(self, freqArray, sr='None', xlength='None'):
        ''' generates an array of sin waves with frequency = "freq"
            optional sr parameter: default = obj global sr
            usage:
            >>tbSigGen = sigGen.odmkSigGen1(numSamples, sr)
            >>testFreqArray = 5000.0
            >>sin5K = tbSigGen.monosin(testFreq2) '''

        if len(freqArray) <= 1:
            print('ERROR (multiSin): freq must be a list of frequencies')
            return 1

        if sr != 'None':
            msinsr = sr
        else:
            msinsr = self.sr
            
        if xlength != 'None':
            xSamples = int(np.ceil(xlength * msinsr))
        else:
            xSamples = self.sampleLength

        # sample period
        T = 1.0 / msinsr

        # create composite sin source
        x = np.linspace(0.0, xSamples*T, xSamples)
        multiSin = 0
        for i in range(xSamples):
            for j in range(len(freqArray)):
                multiSin = multiSin + np.sin(freqArray[j] * 2.0*np.pi * x[i])
            multiSin = (0.999 / max(multiSin)) * multiSin
            yield multiSin



    def multiSinArray(self, freqArray, sr='None', xlength='None'):
        ''' generates an array of sin waves with frequency = "freq"
            optional sr parameter: default = obj global sr
            usage:
            >>tbSigGen = sigGen.odmkSigGen1(numSamples, sr)
            >>testFreqArray = 5000.0
            >>sin5K = tbSigGen.monosin(testFreq2) '''

        if len(freqArray) <= 1:
            print('ERROR (multiSin): freq must be a list of frequencies')
            return 1

        if sr != 'None':
            msinsr = sr
        else:
            msinsr = self.sr
            
        if xlength != 'None':
            xSamples = int(np.ceil(xlength * msinsr))
        else:
            xSamples = self.sampleLength

        # sample period
        T = 1.0 / msinsr

        # create composite sin source
        x = np.linspace(0.0, xSamples*T, xSamples)
        multiSin = np.zeros([xSamples])
        for i in range(len(freqArray)):
            for j in range(xSamples):
                curr = multiSin[j]
                multiSin[j] = curr + np.sin(freqArray[i] * 2.0*np.pi * x[j])
        multiSin = (0.999 / max(multiSin)) * multiSin
        return multiSin



    # // *-----------------------------------------------------------------* //
    # // *---gen simple periodic triangle waveforms (sigLength # samples)
    # // *-----------------------------------------------------------------* //


    # *** FIXIT - need to fix gain scale value (not correct esp. @ fast freqs)

    def monotri(self, freq, sr='None', xlength='None'):
        ''' generates mono triangle wav with frequency = "freq"
            optional sr parameter: default = obj global sr
            usage:
            >>tbSigGen = sigGen.odmkSigGen1(numSamples, sr)
            >>testFreq = 5000.0
            >>tri5K = tbSigGen.monotri(testFreq) '''
            
            
        if sr != 'None':
            trisr = sr
        else:
            trisr = self.sr
            
        if xlength != 'None':
            xSamples = int(np.ceil(xlength * trisr))
        else:
            xSamples = self.sampleLength
        
        T = 1 / trisr
        
        Tfreq = 1 / freq
        TQtrFreq = (1 / (freq * 4))
        
        cycleSamples = Tfreq / T
        qtrCycleSamples = TQtrFreq / T
        
        # determine gain to scale output signal (-1 <-> +1)
        scale = 1 / qtrCycleSamples
        

        #pdb.set_trace()
        # create a triangle signal
        monotri = 0
        for i in range(xSamples):
            j = (i) % cycleSamples
            if (j < qtrCycleSamples) or (j >= (qtrCycleSamples * 3)):
                monotri += 1
                yield monotri * scale
            elif (j >= qtrCycleSamples) or (j < (qtrCycleSamples * 3)):
                monotri -= 1
                yield monotri * scale


    def monotriArray(self, freq, sr='None', xlength='None'):
        ''' generates mono triangle wav with frequency = "freq"
            optional sr parameter: default = obj global sr
            usage:
            >>tbSigGen = sigGen.odmkSigGen1(numSamples, sr)
            >>testFreq = 5000.0
            >>tri5K = tbSigGen.monotri(testFreq) '''
            
            
        if sr != 'None':
            trisr = sr
        else:
            trisr = self.sr
            
        if xlength != 'None':
            xSamples = int(np.ceil(xlength * trisr))
        else:
            xSamples = self.sampleLength
        
        T = 1 / trisr
        
        Tfreq = 1 / freq
        TQtrFreq = (1 / (freq * 4))
        
        cycleSamples = Tfreq / T
        qtrCycleSamples = TQtrFreq / T
        
        # determine gain to scale output signal (-1 <-> +1)
        scale = 1 / qtrCycleSamples

        # pdb.set_trace()
        # create a triangle signal
        currentAmp = 0
        monotri = np.array([])
        for i in range(xSamples):
            #j = (i+1) % cycleSamples
            j = (i) % cycleSamples
            if (j < qtrCycleSamples) or (j >= (qtrCycleSamples * 3)):
                currentAmp += 1
                monotri = np.append(monotri, currentAmp)
            elif (j >= qtrCycleSamples) or (j < (qtrCycleSamples * 3)):
                currentAmp -= 1
                monotri = np.append(monotri, currentAmp)
        monotri = monotri * scale
        return monotri


    # //////////////////////////////////////////////////////////////
    # begin : Wave Table Oscillator Function
    # \\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\

    
    
    def tablegen(self, shape, depth):        
        ''' create look-up table entries for different waveforms

            shape: 
            1=sin,     2=cos,     3=tri,     4=saw-up,    5=saw-dn,
            6=exp-up,  7=exp-dn,  8=log-up,  9=log-dn,    10=cheby,
            11=pulse1, 12=pulse2, 13=pulse3, 14=pulse4,   15=user
        '''

        #wavDict = {0: "sin", 1: "sin", 2: "cos", 3: "tri", 4: "saw-up", 
        #           5: "saw-dn", 6: "exp-up", 7: "exp-dn", 8: "log-up", 
        #           9: "log-dn", 10: "cheby", 11: "pulse1", 12: "pulse2", 
        #           13: "pulse3", 14: "user"}
        wavDict = {"def": 0, "sin": 1, "cos": 2, "tri": 3, "saw-up": 4, 
                   "saw-dn": 5, "exp-up": 6, "exp-dn": 7, "log-up": 8, 
                   "log-dn": 9, "cheby": 10, "pulse1": 11, "pulse2": 12,
                   "pulse3": 13, "pulse4": 14, "user": 15}
        wavShape = wavDict[shape]

        table1 = np.zeros(depth)
        
        if wavShape == 1:    # store 1 cycle of sin
            t1 = np.linspace(0.0, 1.0, depth+1)
            t1 = t1[0:depth]          # crops off the last point for proper cyclic table
            for q in range(depth):
                table1[q] = np.sin(2 * np.pi*(t1[q]))
                
        if wavShape == 2:    # store 1 cycle of cos
            t1 = np.linspace(0.0, 1.0, depth+1)
            t1 = t1[0:depth]          # crops off the last point for proper cyclic table
            for q in range(depth):
                table1[q] = np.cos(2 * np.pi*(t1[q]))                

        elif wavShape == 3:    # store 1 cycle of tri (4 cases to handle arbitrary depth)
            if depth % 4 == 0:
                # first quarter cycle + 1
                table1[0:round(depth/4)+1] = np.linspace(0,1,round(depth/4)+1)
                # 2nd & 3rd quarter cycles +1 (overwrite last value of previous w/1)
                table1[round(depth/4):3*(round(depth/4))+1] = np.linspace(1,-1,2*(round(depth/4))+1)
                triQtrTmp = np.linspace(-1,0,round(depth/4)+1)
                table1[3*(round(depth/4)):depth] = triQtrTmp[0:len(triQtrTmp)-1]
            elif depth % 4 == 1:
                table1[0:round(depth/4)+1] = np.linspace(0,1,round(depth/4)+1)
                table1[round(depth/4):3*(round(depth/4))+1] = np.linspace(1,-1,2*(round(depth/4))+1)
                triQtrTmp = np.linspace(-1,0,round(depth/4)+2)
                table1[3*(round(depth/4)):depth] = triQtrTmp[0:len(triQtrTmp)-1]
            elif depth % 4 == 2:
                table1[0:round(depth/4)] = np.linspace(0,1,round(depth/4))
                table1[round(depth/4)-1:3*(round(depth/4))-1] = np.linspace(1,-1,2*(round(depth/4)))
                triQtrTmp = np.linspace(-1,0,round(depth/4)+1)
                table1[3*(round(depth/4))-2:depth] = triQtrTmp[0:len(triQtrTmp)-1]
            elif depth % 4 == 3:
                table1[0:round(depth/4)+1] = np.linspace(0,1,round(depth/4)+1)
                table1[round(depth/4):3*(round(depth/4))+1] = np.linspace(1,-1,2*(round(depth/4))+1)
                triQtrTmp = np.linspace(-1,0,round(depth/4))
                table1[3*(round(depth/4)):depth] = triQtrTmp[0:len(triQtrTmp)-1]

        elif wavShape == 4:    # store 1 cycle of saw-up
            table1 = np.linspace(-1,1,depth)

        elif wavShape == 5:    # store 1 cycle of saw-down
            table1 = np.linspace(1,-1,depth)
            
        elif wavShape == 6:    # store 1 cycle of exp-up
            tmp1 = np.linspace(0,4,depth)
            tmp1 = np.exp(tmp1)
            table1 = (1/max(tmp1))*tmp1
            
        elif wavShape == 7:    # store 1 cycle of exp-down
            tmp1 = np.linspace(4,0,depth)
            tmp1 = np.exp(tmp1)
            table1 = (1/max(tmp1))*tmp1
            
        elif wavShape == 8:    # store 1 cycle of log-up
            tmp1 = np.linspace(1,64,depth)
            tmp1 = np.log(tmp1)
            table1 = (1/max(tmp1))*tmp1
            
        elif wavShape == 9:    # store 1 cycle of log-down
            tmp1 = np.linspace(64,1,depth)
            tmp1 = np.log(tmp1)
            table1 = (1/max(tmp1))*tmp1

        elif wavShape == 10:    # store 1 cycle of chebychev
            t1 = np.linspace(0.0, 1.0, depth+1)
            t1 = t1[0:depth]          # crops off the last point for proper cyclic table
            for r in range(len(t1)):
                table1[r] = np.cos(13 * np.arccos(t1[r]))

        elif wavShape == 11:    # store 1 cycle of pulse1
            t2 = np.linspace(1,0,depth)**3
            for s in range(len(t2)):
                table1[s] = np.sin(5 * np.pi*(t2[s]))

        elif wavShape == 12:    # store 1 cycle of pulse2
            t2 = np.linspace(1,0,depth)**3
            for s in range(len(t2)):
                table1[s] = np.sin(9 * np.pi*(t2[s]))

        elif wavShape == 13:    # store 1 cycle of pulse3
            t2 = np.linspace(1,0,depth)**3        
            for s in range(len(t2)):
                table1[s] = np.sin(23 * np.pi*(t2[s]))

        elif wavShape == 14:    # store 1 cycle of pulse4
            # create a pseudo-symmetrical exponetial pulse
            # crops off the last point for proper cyclic table
            t3_1 = np.linspace(0,1,int(np.floor(depth/2)+1))
            t3_2 = np.linspace(1,0,int(np.ceil(depth/2)+1))
            t3 = np.concatenate(( t3_1[0:len(t3_1)-1], t3_2[0:len(t3_2)-1] ))**3        
            for t in range(t3):
                table1[t] = np.cos(5 * np.pi*(t3[t]))
                
        elif wavShape == 15:
            # **FIXIT** - implement user-input wavetable data
            t1 = np.linspace(0.0, 1.0, depth+1)
            t1 = t1[0:depth]          # crops off the last point for proper cyclic table
            for q in range(depth):
                table1[q] = np.sin(2 * np.pi*(t1[q]))                

        else:    # default
            t1 = np.linspace(0.0, 1.0, depth+1)
            t1 = t1[0:depth]          # crops off the last point for proper cyclic table
            for q in range(depth):
                table1[q] = np.sin(2 * np.pi*(t1[q]))
        
        return table1

        
        
    # wavetable oscillator function

    def xodWTOsc(self, numSamples, shape, freqCtrl, phaseCtrl, quant='None'):
        
        ''' *--------------------------------------------------------*   
        # xodWTOsc: single channel wavetable oscillator
        #
        # shape:
        # <<sin, cos, tri, saw-up, saw-dn, exp-up, exp-dn, log-up, log-dn>>
        #
        # The output frequency can be fixed or variable
        # When freqCtrl is a single scalar value, the output freq. is fixed
        # When freqCtrl is an array of length=numSamples, the output freq. varies each step
        #
        # The output phase can be fixed or variable
        # When phaseCtrl is a single scalar value, the output phase is fixed
        # When phaseCtrl is an array of length=numSamples, the output phase varies each step 
        #
        # If quant != None, quantize output to integer range +/- quant
        #
        # *--------------------------------------------------------* // '''
    
        tableDepth = 4096    

        tb = self.tablegen(shape,tableDepth);

        Fs = self.sr

        # sample period
        #T = 1.0 / Fs

        accWidth = 48
        qntWidth = np.ceil(np.log2(tableDepth))
        lsbWidth = accWidth - qntWidth    # expect 36
        lsbWidthScale = 2**lsbWidth
        lsbWidthUnScale = 2**-lsbWidth

        if not( isinstance(freqCtrl,int) or isinstance(freqCtrl,float) or isinstance(freqCtrl,list) or isinstance(freqCtrl, np.ndarray)):
            print('ERROR (xodWTOsc): freqCtrl must be a single freq value, a list, or a numpy array of frequency values')
            return 1            
        elif ( (isinstance(freqCtrl,list) or isinstance(freqCtrl, np.ndarray)) and len(freqCtrl) < numSamples ):
            print('ERROR (xodWTOsc): freqCtrl array must be at least numSamples long')
            return 1
        elif (isinstance(freqCtrl,int) or isinstance(freqCtrl,float)):    # fixed freq
            # scale freq to match quantizer
            freqCtrlScale = lsbWidthScale * freqCtrl
            skipInc = ((2**qntWidth) * freqCtrlScale) / Fs
        elif (isinstance(freqCtrl,list) or isinstance(freqCtrl, np.ndarray)):     # variable freq
            freqCtrlScale = lsbWidthScale * freqCtrl[0]           
            skipInc = ((2**qntWidth) * freqCtrlScale) / Fs

        if not( isinstance(phaseCtrl,int) or isinstance(phaseCtrl,float) or isinstance(phaseCtrl,list) or isinstance(phaseCtrl, np.ndarray)):        
            print('ERROR (xodWTOsc): phaseCtrl must be a single phase value, or an array of phase values')
            return 1
        elif ( (isinstance(phaseCtrl,list) or isinstance(phaseCtrl, np.ndarray)) and len(phaseCtrl) < numSamples ):
            print('ERROR (xodWTOsc): phaseCtrl array must be at least numSamples long')
            return 1
        # initialize variables so that the OSC starts at the beginning of the table            
        elif (isinstance(phaseCtrl,int) or isinstance(phaseCtrl,float)):    # fixed phase            
            # scale phase offset
            # converts radians into a scaled phase offset to be added to the output of the acc
            # dependent on the table depth
            phaseOffset = round( ((phaseCtrl * tableDepth) / (2 * np.pi)) % tableDepth )
            accAddr = phaseOffset * lsbWidthScale    # init to 'zero plus phase shift'
        elif (isinstance(phaseCtrl,list) or isinstance(phaseCtrl, np.ndarray)):
            phaseOffset = round( ((phaseCtrl[0] * tableDepth) / (2 * np.pi)) % tableDepth )
            accAddr = phaseOffset * lsbWidthScale    # init to 'zero plus phase shift'

        # ***initialize***

        # init osc output vectors
        xodOsc = np.zeros([numSamples])
        xodOsc90 = np.zeros([numSamples])
        xodSqrPulse = np.zeros([numSamples])
            

        # used to add a 90 deg offset for complex sinusoid generation
        offset90 = tableDepth / 4

        accAddr90 = (offset90 + phaseOffset) * lsbWidthScale    # init to 'zero plus offset90 plus phase shift'
        qntAddr = int(phaseOffset)
        qntAddr90 = int(qntAddr + offset90)
        yLow = tb[qntAddr]
        yLow90 = tb[qntAddr90]

        # **main loop**
        # ___::((Interpolated WaveTable))::___
        # generates main osc waveform out, 90deg shifted out (sin/cos), square pulse out
    
        for i in range(numSamples):    # osc main loop
            
            if (isinstance(freqCtrl,list) or isinstance(freqCtrl, np.ndarray)):
                freqCtrlScale = lsbWidthScale * freqCtrl[i]    # scalar or array

            accAddrP1 = accAddr + lsbWidthScale
            accAddr90P1 = accAddr90 + lsbWidthScale
    
            # \\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\
            # generate oscillator output
            # ///////////////////////////////////////////////////        
            yHigh = tb[(qntAddr + 1)%tableDepth]
            yHigh90 = tb[(qntAddr90 + 1)%tableDepth]
            xodOsc[i] = yLow + (yHigh-yLow) * ((accAddrP1 - (qntAddr * lsbWidthScale)) * lsbWidthUnScale)
            xodOsc90[i] = yLow90 + (yHigh90 - yLow90) * ((accAddr90P1 - (qntAddr90 * lsbWidthScale)) * lsbWidthUnScale)

            if quant!='None':
                if isinstance(quant, int):
                    #xodOscQuant[i] = int(round(quant*odmkOsc[i]))
                    xodOsc[i] = int(round(quant*xodOsc[i]))
                    xodOsc90[i] = int(round(quant*xodOsc90[i]))
                else:
                    print('quantization value must be an integer')
                    return 1


            # generate square pulse output
            if xodOsc[i] >= 0:
                xodSqrPulse[i] = 1
            else:
                xodSqrPulse[i] = 0
  
            # phase accumulator
            #accAddr = (accAddr + skipInc[i]) % 2**accWidth
            #accAddr90 = (accAddr90 + skipInc[i]) % 2**accWidth
            accAddr = (accAddr + skipInc) % 2**accWidth
            accAddr90 = (accAddr90 + skipInc) % 2**accWidth            
    
            # quantize
            qntAddr = int(np.floor(accAddr * lsbWidthUnScale))
            qntAddr90 = int(np.floor(accAddr90 * lsbWidthUnScale))
    
            yLow = tb[qntAddr]
            yLow90 = tb[qntAddr90]
            # temp
            #yLow_tap[i] = yLow
            #yLow90_tap[i] = yLow90
            
        return xodOsc, xodOsc90, xodSqrPulse
 
        #if quant!='None':
        #    return odmkOscQuant
        #else:
        #    return odmkOsc, odmkOsc90, odmkSqrPulse
        
        
        


    # #########################################################################
    # begin : waveform generators
    # #########################################################################

    phaseAcc = 0

    def pulseWidthMod(self, phaseInc, pulseWidth):
        ''' generates pulse width modulated square wav
            usage:
            >>phaseInc = output base freq - Fo / Fs
            >>pulseWidth = (% of cycle)/100 -> [0 - 1] '''

        # create pulse width modulated square wave
        if self.phaseAcc >= 1.0:
            self.phaseAcc -= 1
        if self.phaseAcc > pulseWidth:
            pwm = 0
        else:
            pwm = 1
            
        self.phaseAcc += phaseInc

        return pwm



    # #########################################################################
    # begin : file output
    # #########################################################################

    # // *-----------------------------------------------------------------* //
    # // *---TXT write simple periodic waveforms (sigLength # samples)
    # // *-----------------------------------------------------------------* //

    def sig2txt(self, sigIn, nChan, outNm, outDir='None'):
        ''' writes data to TXT file
            signal output name = outNm (expects string) '''

        if outDir != 'None':
            try:
                if isinstance(outDir, str):
                    txtOutDir = outDir
                    os.makedirs(txtOutDir, exist_ok=True)
            except NameError:
                print('Error: outNm must be a string')
        else:
            txtOutDir = self.sigGenOutDir

        try:
            if isinstance(outNm, str):
                sigOutFull = txtOutDir+outNm
        except NameError:
            print('Error: outNm must be a string')

        # writes data to .TXT file:
        outputFile = open(sigOutFull, 'w', newline='')

        if nChan == 0:
            print('ERROR: Number of Channels must be >= 1')
        elif nChan == 1:
            for i in range(len(sigIn)):
                outputFile.write(str(sigIn[i]) + '\n')
        else:
            for i in range(len(sigIn[0])):
                lineTmp = ""
                for j in range(len(sigIn) - 1):
                    strTmp = str(sigIn[j, i]) + str('    ')
                    lineTmp = lineTmp + strTmp
                lineTmp = lineTmp + str(sigIn[len(sigIn) - 1, i]) + '\n'
                outputFile.write(lineTmp)            

        outputFile.close()
        

    # // *-----------------------------------------------------------------* //
    # // *---CSV write simple periodic waveforms (sigLength # samples)
    # // *-----------------------------------------------------------------* //

    def sig2csv(self, sigIn, outNm, outDir='None'):
        ''' writes data to CSV file
            signal output name = outNm (expects string) '''

        if outDir != 'None':
            try:
                if isinstance(outDir, str):
                    csvOutDir = outDir
                    os.makedirs(csvOutDir, exist_ok=True)
            except NameError:
                print('Error: outNm must be a string')
        else:
            csvOutDir = self.sigGenOutDir

        try:
            if isinstance(outNm, str):
                sigOutFull = csvOutDir+outNm
        except NameError:
            print('Error: outNm must be a string')

        # writes data to .CSV file:
        outputFile = open(sigOutFull, 'w', newline='')
        outputWriter = csv.writer(outputFile)

        for i in range(len(sigIn)):
            tmpRow = [sigIn[i]]
            outputWriter.writerow(tmpRow)

        outputFile.close()

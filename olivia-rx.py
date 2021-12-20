#!/usr/bin/python3
# -*- coding: utf-8 -*-
"""
OLIVIA MFSK Modulator
@author: Federico Santandrea

Source code is lightly commented for clarity.
Please read the document to better understand the various processes.
"""

# Libraries
import os, sys
import numpy as np
import sounddevice as sd
from numpy.fft import fft

# Hardcoded parameters
DEFAULT_PARAMS = "32/1000@1500"
SAMPLE_RATE = 8000
BLOCK_THRESHOLD = 24

# Global parameters
'''
Input:
- CENTER_FREQUENCY, SYMBOLS (# of tones), BANDWIDTH
Computed:
- spb: number of bits for a symbol
- fsep: frequency separation between tones, in Hz (also = baud rate)
- wlen: time separation between tones, in samples
'''

# Global objects
'''
- buf: audio sample input buffer
- sin: sounddevice InputStream for sample acquisition
'''

def __main__():
    '''
    Main program flow.
    '''
    welcomeAndSetup()
    initSound()
    
    syms = []

    while True:
        # Fetch new samples
        updateBuffer()
     
        sym = detectSymbol()
        syms.append(sym)
        
        if len(syms) == 64:
            # Enough symbols to decode a block
            if decodeAndPrintBlock(syms):
                # Block decoded successfully, waiting for a new one
                syms = [] 
            else:
                # Probably not a complete block, try rolling
                syms = syms[1:]

#

def welcomeAndSetup():
    '''
    Decodes command line parameters and prints a welcome message.
    '''
    global CENTER_FREQUENCY, SYMBOLS, BANDWIDTH
    global spb, fsep, wlen, slen
    
    if len(sys.argv) == 1:
        params = DEFAULT_PARAMS
    elif len(sys.argv) == 2:
        params = sys.argv[1]
    else:
        printUsageAndQuit()
        
    try:
        CENTER_FREQUENCY = int(params.split("@")[1])
        SYMBOLS = int(params.split("/")[0])
        BANDWIDTH = int(params.split("@")[0].split("/")[1])
        spb = int(np.log2(SYMBOLS))
        fsep = BANDWIDTH/SYMBOLS # = baud
        wlen = int(np.ceil(SAMPLE_RATE/fsep))
        
    except:
        printUsageAndQuit()

    if os.isatty(0):
        print("*** Olivia demodulator ***")
        print("(C) 2020 Federico Santandrea")
        print("federico.santandrea@studio.unibo.it")
        print()
        print("Starting Olivia demodulator at center " + str(CENTER_FREQUENCY) +
              "Hz, using " + str(SYMBOLS) + " tones over " + str(BANDWIDTH) + "Hz")
        print()

def printUsageAndQuit():
    '''
    Prints usage help if needed.
    '''
    print("usage: " + sys.argv[0] + " [syms/bandwidth@centerfrequency]")
    print("Example (default): " + sys.argv[0] + " " + DEFAULT_PARAMS)
    quit()

def initSound():
    '''
    Prepares global InputStream for sample acquisition.
    '''
    global sin, buf
    
    sin = sd.InputStream(samplerate=SAMPLE_RATE, blocksize=wlen,
        #channels=2, device=2, # stereo mix
        dtype=np.float32)
    sin.start()
    buf= np.zeros(wlen)

def updateBuffer():
    '''
    Acquires a new wlen-ful of samples from audio device.
    '''
    global buf
    
    (samples, of) = sin.read(wlen)
    buf = samples[:,0] # consider only one channel

def detectSymbol():
    '''
    Applies Fourier transform to audio buffer to detect
    symbol corresponding to sampled tone.

    Returns
    -------
    int
        Most likely symbol number.
    '''
    spectrum = np.abs(fft(buf))
    ix = CENTER_FREQUENCY - BANDWIDTH/2 + fsep/2
    measures = np.zeros(SYMBOLS)

    for i in range(0, SYMBOLS):
        ix += fsep
        measures[i] = spectrum[int(ix * wlen / SAMPLE_RATE)]
    mix = np.argmax(measures)
    
    return degray(mix)

def decodeAndPrintBlock(syms):
    '''
    Decodes a full block of 64 symbols, then prints it
    to standard output.
    '''
    w = np.zeros((spb, 64))
    
    # key = 0xE257E6D0291574EC
    key = np.flip(np.array(
          [1, 1, 1, 0, 0, 0, 1, 0,
           0, 1, 0, 1, 0, 1, 1, 1,
           1, 1, 1, 0, 0, 1, 1, 0,
           1, 1, 0, 1, 0, 0, 0, 0,
           0, 0, 1, 0, 1, 0, 0, 1,
           0, 0, 0, 1, 0, 1, 0, 1,
           0, 1, 1, 1, 0, 1, 0, 0,
           1, 1, 1, 0, 1, 1, 0, 0]))
    
    output = ""
    doubt = 0
    for i in range(0, spb):
        for j in range(0, 64):
            bit = (syms[j] >> ((i+j) % spb)) & 1
            if bit == 1:
                w[i,j] = -1
            else:
                w[i,j] = 1
                
        w[i,:] = w[i,:] * (-2*np.roll(key, -13*i)+1)
        w[i,:] = fwht(w[i,:])
        
        c = np.argmax(np.abs(w[i,:]))
        
        if abs(w[i,c]) < BLOCK_THRESHOLD:
            doubt += 1
            
        if w[i,c] < 0:
            c = c + 64    
        if c != 0:
            output += chr(c)
    
    if doubt == 0:
        print(output, end="", flush=True)
        return True
    else:
        return False
        
#

def fwht(data):
    '''
    Fast Walsh-Hadamard transform.
    '''
    step = 1
    while step < len(data):
        for ptr in range(0, len(data), 2*step):
            for ptr2 in range(ptr, step+ptr):
                bit1 = data[ptr2]
                bit2 = data[ptr2+step]
                
                newbit1 = bit2
                newbit1 = newbit1 + bit1
                
                newbit2 = bit2
                newbit2 = newbit2 - bit1
                
                data[ptr2] = newbit1
                data[ptr2+step] = newbit2
                
        step *= 2
    return data

def degray(n):
    mask = n
    while mask != 0:
        mask >>= 1
        n ^= mask
    return n

#
#

__main__()

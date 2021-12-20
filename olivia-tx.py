#!/usr/bin/python3
# -*- coding: utf-8 -*-
"""
OLIVIA MFSK Modulator
@author: Federico Santandrea

Source code is lightly commented for clarity.
Please read the document to better understand the various processes.
"""

# Libraries
import os
import sys
from queue import Queue
from time import sleep
import sounddevice as sd
import numpy as np
from numpy import pi

# Hardcoded parameters
SAMPLE_RATE = 8000
ATTENUATION = 30
DEFAULT_PARAMS = "32/1000@1500"
ENABLE_PREAMBLE = True

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
- q: FIFO queue to hold complete blocks for transmission
- sout: sounddevice OutputStream for sample playback
- trail: buffer containing trail of last symbol, for overlapping
'''

def __main__():
    '''
    Main program flow.
    '''
    welcomeAndSetup()
    initSound()

    # Sample generation starts in background, playing null blocks
    # until real data is in the transmission queue.
    
    if os.isatty(0):
        prompt = "olivia> "
    else:
        prompt = ""
            
    while True:
        try:
            text = input(prompt) + "\n"
            
            # Splits text in pieces, padding last one.
            # Then puts the pieces in transmission queue.
            for i in range(0, len(text), spb):
                piece = text[i:i+spb]
                while len(piece) < spb:
                    piece = piece + "\0"
                q.put(piece)
            
        except:
            q.put(None)
            if os.isatty(0):
                print()
                
            while not q.empty():
                sleep(1)
                
            sleep(10)
            quit()

def welcomeAndSetup():
    '''
    Decodes command line parameters and prints a welcome message.
    '''
    global CENTER_FREQUENCY, SYMBOLS, BANDWIDTH
    global spb, fsep, wlen
    global q, trail
    
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
        print("*** Olivia modulator ***")
        print("(C) 2020 Federico Santandrea")
        print("federico.santandrea@studio.unibo.it")
        print()
        print("Starting Olivia modulator at center " + str(CENTER_FREQUENCY) +
              "Hz, using " + str(SYMBOLS) + " tones over " + str(BANDWIDTH) + "Hz")
        print()
    
    q = Queue()
    trail = np.zeros(wlen)
    
def printUsageAndQuit():
    '''
    Prints usage help if needed.
    '''
    print("usage: " + sys.argv[0] + " [syms/bandwidth@centerfrequency]")
    print("Example (default): " + sys.argv[0] + " " + DEFAULT_PARAMS)
    quit()


def initSound():
    '''
    Prepares global OutputStream for sample playback.
    '''
    global sout
    
    sout = sd.OutputStream(samplerate=SAMPLE_RATE, blocksize=64*wlen,
        channels=1, callback=callback, dtype=np.float32)
    sout.start()

def callback(outdata, frames, time, status):
    '''
    Handles sample generation and playback asynchronously.
    '''
    
    # If first call, generate the preamble.
    if callback.firstCall and ENABLE_PREAMBLE:
        callback.firstCall = False
        outdata[:,0] = generatePreamble()/ATTENUATION
        return None
    
    # If last call, just empty the samples buffer.
    if callback.lastCall:
        outdata.fill(0)
        return None
    
    # If transmission queue is empty, transmit null blocks.
    try:
        piece = q.get_nowait()
    except:
        piece = "\0" * spb
        
    outdata[:,0] = generateBlock(piece)/ATTENUATION
    
    # If transmission is over, transmit ending tail.
    if piece == None:
        callback.lastCall = True

    return None

callback.firstCall = True
callback.lastCall = False

def generatePreamble():
    '''
    A preamble is the beginning tail before data transmission.
    But if it doesn't fit in a full block buffer, don't bother.
    '''
    wf = np.zeros(64*wlen)
    tail = generateTail()
    if len(tail) < 64*wlen:
        wf[64*wlen-len(tail):64*wlen] = tail
    return wf
    
def generateTail():
    '''
    A tail is made in this way:
        first tone, last tone, first tone, last tone
    each one lasting 1/4 seconds.
    '''
    pl = int(SAMPLE_RATE/4)
    t = np.arange(0, 1/4, 1/SAMPLE_RATE)
    wf = np.zeros(SAMPLE_RATE)
    wf[0:pl] = toneShaper(np.sin(2*pi*(CENTER_FREQUENCY-BANDWIDTH/2+fsep/2)*t)/2)
    wf[pl:2*pl] = toneShaper(np.sin(2*pi*(CENTER_FREQUENCY+BANDWIDTH/2-fsep/2)*t)/2)
    wf[2*pl:3*pl] = wf[0:pl]
    wf[3*pl:4*pl] = wf[pl:2*pl]
    return wf
    
def generateBlock(piece):
    '''
    Transmits samples corresponding to a full block.
    '''
    global trail
    
    wf = np.zeros(64*wlen+wlen)
    
    # Overlaps trail of last symbol, if any
    wf[0:wlen] += trail
    
    # If transmission is being stopped, add trailing tail
    if piece == None:
        if not ENABLE_PREAMBLE:
            return wf[0:64*wlen]
        trail = np.zeros(wlen)
        tail = generateTail()
        if len(tail) < 64*wlen:
            wf[wlen:wlen+len(tail)] = tail
        return wf[0:64*wlen]
    
    syms = prepareSymbols(piece)
    
    for i in range(0, 64):
        # Tone number is symbol number after Gray encoding
        # This minimized error made by mistaking one tone for
        # another one right next to it (1 wrong bit only).
        tone = oliviaTone(gray(syms[i]))
        wf[wlen*i:wlen*i+len(tone)] += tone
        
    trail = wf[64*wlen:64*wlen+wlen]
    return wf[0:64*wlen]

def oliviaTone(toneNumber):
    '''
    Tone generator. Creates output waveform for specified tone number.
    '''
    toneFreq = (CENTER_FREQUENCY - BANDWIDTH/2) + fsep/2 + fsep * toneNumber
    t = np.arange(0, 2/fsep, 1/SAMPLE_RATE)
    ph = np.random.choice([-pi/2, pi/2]);
    ret = np.sin(2*pi*toneFreq*t + ph)
    return toneShaper(ret)

def toneShaper(toneData):
    '''
    Tone shaping to avoid intersymbol modulation.
    Cosine coefficients are fixed and can be found in specification.
    '''
    x = np.linspace(-pi, pi, len(toneData));
    shape = (1. + 1.1913785723*np.cos(x)
             - 0.0793018558*np.cos(2*x)
             - 0.2171442026*np.cos(3*x)
             - 0.0014526076*np.cos(4*x))
    return toneData * shape

#

def prepareSymbols(chars):
    '''
    Transform a block of characters into a block of symbols
    ready for transmission.
    '''
    w = np.zeros((spb, 64))
    
    # Key is a 64-bit fixed value and can be found in specification.
    #   key = 0xE257E6D0291574EC
    # It is a pseudorandom value and its role is to make the
    # output stream appear random.
    # Here it is decomposed in a bit array for easier use.
    key = np.flip(np.array(
          [1, 1, 1, 0, 0, 0, 1, 0,
           0, 1, 0, 1, 0, 1, 1, 1,
           1, 1, 1, 0, 0, 1, 1, 0,
           1, 1, 0, 1, 0, 0, 0, 0,
           0, 0, 1, 0, 1, 0, 0, 1,
           0, 0, 0, 1, 0, 1, 0, 1,
           0, 1, 1, 1, 0, 1, 0, 0,
           1, 1, 1, 0, 1, 1, 0, 0]))
    
    # Character to 64-value vector mapping to provide redundancy.
    # Characters are 7-bit (value from 0 to 127)
    # Values from 0 to 63 are mapped by setting nth value to 1
    # Values from 64 to 127 are mapped by setting nth value to -1
    # Every other value in vector is 0.
    for i in range(0, spb):
        q = ord(chars[i])
        if q > 127:
            q = 0

        if q < 64:
            w[i, q] = 1
        else:
            w[i, q-64] = -1
        
        # Inverse Walsh-Hadamard Transform to encode redundancy
        w[i,:] = ifwht(w[i,:])
        
        # XOR with key to ensure randomness
        # (XOR is made by multiplying with -1 or 1)
        w[i,:] = w[i,:] * (-2*np.roll(key, -13*i)+1)
    
    syms = np.zeros((64, spb))
    
    # Bit interleaving to spread errors over symbols
    for bis in range(0, spb):
        for sym in range(0, 64):
            q = 100*spb + bis - sym
            if w[q % spb,sym] < 0:
                syms[sym,bis] = 1

    # Convert to integer to find symbol numbers
    symn = np.zeros(64)
    for i in range(0,64):
        symn[i] = bits2int(np.flip(syms[i]))

    return symn

#

def ifwht(data):
    '''
    Inverse Fast Walsh-Hadamard transform.
    There is a similar ready-made transform in sympy, but its
    output ordering (Hadamard order) is different from the Olivia
    specified one, and it's more efficient to reimplement it
    directly rather than converting the output.
    
    This is a 1:1 translation from the Olivia C++ reference
    implementation.
    '''
    step = int(len(data)/2)
    while step >= 1:
        for ptr in range(0, 64, 2*step):
            for ptr2 in range(ptr, step+ptr):
                bit1 = data[ptr2]
                bit2 = data[ptr2+step]
                
                newbit1 = bit1
                newbit1 = newbit1 - bit2
                newbit2 = bit1
                newbit2 = newbit2 + bit2
                
                data[ptr2] = newbit1
                data[ptr2+step] = newbit2
        step = int(step/2)
    return data

def bits2int(A):
    '''
    Utility function to transform a bit array to an integer.
    '''
    return int(str(A).replace(".", "").replace(",", "").replace(" ", "")
     .replace("[", "").replace("]", ""), 2)

def gray(n):
    '''
    Utility function to calculate Gray encoding of an integer value
    '''
    n = int(n)
    return n ^ (n >> 1)

#
#

__main__()

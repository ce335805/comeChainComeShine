import numpy as np
import globalSystemParams as prms
import random


def getInitialStateConst():
    initialState = np.zeros(prms.chainLength, dtype='double') + .5
    return initialState


def getInitialStateShuffle():
    random.seed(13)
    initialState = np.zeros(prms.chainLength, dtype='double')
    initialState[: prms.numberElectrons] = 1.
    random.shuffle(initialState)
    #print(initialState)
    return initialState


def getG0InitialStateShift(initialShift):
    initialState = np.zeros(prms.chainLength, dtype='double')
    initialState[ : prms.numberElectrons // 2 + 1 + initialShift] = 1.0
    initialState[ prms.chainLength - prms.numberElectrons//2 + initialShift: ] = 1.0
    return initialState

def getG0InitialStateAna():
    initialState = np.zeros(prms.chainLength + 1, dtype='double')
    initialShift = 0
    initialState[ : prms.numberElectrons // 2 + 1 + initialShift] = 1.0
    initialState[ prms.chainLength - prms.numberElectrons//2 + initialShift: -1] = 1.0
    return initialState
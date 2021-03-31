import numpy as np
import globalSystemParams as prms


def getG0InitialStateNum():
    initialState = np.zeros(prms.chainLength, dtype='double')
    initialState[ : prms.numberElectrons // 2 + 1] = 1.0
    initialState[ prms.chainLength - prms.numberElectrons//2: ] = 1.0
    return initialState

def getG0InitialStateAna():
    initialState = np.zeros(prms.chainLength + 1, dtype='double')
    initialState[ : prms.numberElectrons // 2 + 1] = 1.0
    initialState[ prms.chainLength - prms.numberElectrons//2: -1] = 1.0
    return initialState
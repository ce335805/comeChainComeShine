import numpy as np
import globalSystemParams as prms
from scipy.special import factorial

import scipy.linalg

def getCoherentStateForN(N):
    #alpha = np.sqrt(N)
#
    #a = np.diag(np.sqrt(np.arange((prms.maxPhotonNumber - 1)) + 1), +1)
    #aDagger = np.diag(np.sqrt(np.arange((prms.maxPhotonNumber - 1)) + 1), -1)
#
    #mat = alpha * aDagger - alpha * a
    #expMat = scipy.linalg.expm(mat)
#
    #ptZeroState = np.zeros((prms.maxPhotonNumber), dtype=complex)
    #ptZeroState[0] = 1.
#
    #cohState = np.dot(expMat, ptZeroState)

    cohState = np.zeros((prms.maxPhotonNumber), dtype=complex)
    index = np.arange(prms.maxPhotonNumber)
    for i in index:
        cohState[i] = (-1)**i * np.sqrt(np.exp(-N) * N**i / factorial(i))

    return cohState
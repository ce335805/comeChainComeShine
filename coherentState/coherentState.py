import numpy as np
import globalSystemParams as prms
from scipy.special import factorial

import scipy.linalg as sciLin

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
        cohState[i] = np.sqrt(np.exp(-N) * N**i / factorial(i))

    return cohState

def getSqueezedState(eta, T):
    aDag = aDagOp()
    a = aOp()

    aDagSq = np.matmul(aDag, aDag)
    aSq = np.matmul(a, a)

    zeta = .25 * np.log(1. - 2. * eta**2 / prms.w0 * T)
    #zeta = .5 * eta**2 / prms.w0

    operator = sciLin.expm(.5 * zeta * (aDagSq - aSq))

    ptState = np.zeros(prms.maxPhotonNumber, dtype=complex)
    ptState[0] = 1.

    ptState = np.dot(operator, ptState)

    return ptState


def aDagOp():
    return np.diag(np.sqrt(np.arange((prms.maxPhotonNumber - 1)) + 1), -1)

def aOp():
    return np.diag(np.sqrt(np.arange((prms.maxPhotonNumber - 1)) + 1), +1)
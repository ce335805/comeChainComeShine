import numpy as np

import globalSystemParams as prms
import scipy.linalg as sciLin

def getExactGS(eta):

    aDag = aDagOp()
    a = aOp()

    aDagSq = np.matmul(aDag, aDag)
    aSq = np.matmul(a, a)

    fac = .125 * np.log(1 + 4 * eta**2 / prms.w0)

    operator = sciLin.expm(fac * (aDagSq - aSq))

    ptState = np.zeros(prms.maxPhotonNumber, dtype=complex)
    ptState[0] = 1.

    ptState = np.dot(operator, ptState)

    return ptState


def xVar(eta):

    ptState = getExactGS(eta)

    x = aDagOp() + aOp()
    xSq = np.matmul(x, x)

    xExpected = np.dot(np.conj(ptState), np.dot(x, ptState))
    xSqExpected = np.dot(np.conj(ptState), np.dot(xSq, ptState))

    return xSqExpected - xExpected**2


def pVar(eta):
    ptState = getExactGS(eta)

    p = aDagOp() - aOp()
    pSq = np.matmul(p, p)

    pExpected = np.dot(np.conj(ptState), np.dot(p, ptState))
    pSqExpected = np.dot(np.conj(ptState), np.dot(pSq, ptState))

    return pSqExpected - pExpected ** 2

def aDagOp():
    return np.diag(np.sqrt(np.arange((prms.maxPhotonNumber - 1)) + 1), -1)

def aOp():
    return np.diag(np.sqrt(np.arange((prms.maxPhotonNumber - 1)) + 1), +1)
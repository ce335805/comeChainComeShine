import numpy as np
import globalSystemParams as prms
from scipy.special import factorial
import scipy.linalg as sciLin
from arb_order import photonState as phState

import energyFunctions as eF
from arb_order import arbOrder

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

    operator = sciLin.expm(-.5 * zeta * (aDagSq - aSq))

    ptState = np.zeros(prms.maxPhotonNumber, dtype=complex)
    ptState[0] = 1.

    ptState = np.dot(operator, ptState)

    return ptState

def getShiftedGS(eta, N):
    alpha = np.sqrt(N)

    a = np.diag(np.sqrt(np.arange((prms.maxPhotonNumber - 1)) + 1), +1)
    aDagger = np.diag(np.sqrt(np.arange((prms.maxPhotonNumber - 1)) + 1), -1)

    mat = alpha * aDagger - alpha * a
    expMat = sciLin.expm(mat)

    ptGS = getPhGS(eta)

    cohState = np.dot(expMat, ptGS)

    return cohState

def aDagOp():
    return np.diag(np.sqrt(np.arange((prms.maxPhotonNumber - 1)) + 1), -1)

def aOp():
    return np.diag(np.sqrt(np.arange((prms.maxPhotonNumber - 1)) + 1), +1)

def gsEffectiveKineticEnergy(eta):
    gsT = - 2. / np.pi * prms.chainLength
    ptState = getSqueezedState(eta, gsT)
    cosOp = sciLin.cosm(eta * (aDagOp() + aOp()))
    kineticE = gsT * np.dot(np.conj(ptState), np.dot(cosOp, ptState))
    return kineticE / prms.chainLength

def gsEffectiveKineticEnergyAnalytical(eta):
    gsT = - 2. / np.pi * prms.chainLength
    fac = np.sqrt(1 - 2. * eta**2 / prms.w0 * gsT)
    return gsT / prms.chainLength * (1 - eta**2 / 2. * fac)

def gsEffectiveKineticEnergyArray(etaArr):
    gsKinetics = np.zeros(len(etaArr))
    for etaInd, eta in enumerate(etaArr):
        gsKinetics[etaInd] = np.real(gsEffectiveKineticEnergy(eta))
        #gsKinetics[etaInd] = np.real(gsEffectiveKineticEnergyAnalytical(eta))
    return gsKinetics


def getPhGS(eta):
    gs = arbOrder.findGS(eta, 3)
    gsJ = eF.J(gs)
    gsT = eF.T(gs)
    return phState.findPhotonGS([gsT, gsJ], eta, 3)

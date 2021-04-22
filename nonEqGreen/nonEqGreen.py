import numpy as np
import sec_order.analyticalEGS as anaGS
import globalSystemParams as prms
import energyFunctions as eF
from arb_order import photonState as phState
from arb_order import numHamiltonians as numH
import scipy.linalg as sciLin
import fourierTrafo as FT
from arb_order import arbOrder
from coherentState import coherentState
import time
from scipy.linalg import expm_frechet
import utils


def gfPointTCoh(kVec, tRel, tAv, eta, N):
    phState = coherentState.getCoherentStateForN(N)
    #phState = coherentState.getSqueezedStateFor(N)
    H = getH(eta)

    print("kVec-Val = {}".format(kVec[0] / np.pi))

    gk = - 2. * prms.t * np.sin(kVec)
    eK = 2. * prms.t * np.cos(kVec)
    x = eta * (np.diag(np.sqrt(np.arange((prms.maxPhotonNumber - 1)) + 1), -1) + np.diag(
        np.sqrt(np.arange((prms.maxPhotonNumber - 1)) + 1), +1))
    sinX = sciLin.sinm(x)
    cosX = sciLin.cosm(x)

    timeStart = time.time()
    iHTRelMTAv, iHTRelPTAv, iHtSinCosTRel = setUpMatricies(H, cosX, eK, gk, sinX, tAv, tRel)
    timeStop = time.time()
    print("setUpMatricies take: {}".format(timeStop - timeStart))

    timeStart = time.time()
    expiHt, expiHtDash, exptRel = setUpExponentialMatricies(iHTRelMTAv, iHTRelPTAv, iHtSinCosTRel, kVec, tAv, tRel)
    timeStop = time.time()
    print("setUpExponentialMatricies take: {}".format(timeStop - timeStart))

    timeStart = time.time()
    GF = evaluateBraKet(expiHt, expiHtDash, exptRel, kVec, phState, tAv, tRel)
    timeStop = time.time()
    print("evaluateBraKet take: {}".format(timeStop - timeStart))


    return - 1j * GF

def gfCohExpDampingGreater(kVec, tRel, tAv, eta, damping, N):
    gsOcc = np.zeros(prms.chainLength, dtype=complex)
    gsOcc[prms.numberElectrons // 2 : - prms.numberElectrons // 2] = 1.
    print(gsOcc)
    dampExptRel = np.exp(- damping * np.abs(tRel))
    GF = gfPointTCoh(kVec, tRel, tAv, eta, N)
    GFDamp = GF[:, :, :] * dampExptRel[None, :, None]
    GFOcc = GFDamp[:, :, :] * gsOcc[:, None, None]
    return GFOcc

def gfCohExpDampingLesser(kVec, tRel, tAv, eta, damping, N):
    #gsOcc = np.zeros(prms.chainLength, dtype='double')
    #gsOcc[ : prms.numberElectrons // 2 + 1] = 1.
    #gsOcc[-prms.numberElectrons // 2 + 1 : ] = 1.
    #print(gsOcc)
    dampExptRel = np.exp(- damping * np.abs(tRel))
    GF = gfPointTCoh(kVec, tRel, tAv, eta, N)
    GFDamp = GF[:, :, :] * dampExptRel[None, :, None]
    GFOcc = GFDamp
    #GFOcc = GFDamp[:, :, :] * gsOcc[:, None, None]
    return GFOcc

def gfCohWGreater(kVec, wRel, tAv, eta, damping, N):
    tRel = FT.tVecFromWVec(wRel)
    tRelPos = tRel[len(tRel) // 2 : ]
    GFT = gfCohExpDampingGreater(kVec, tRelPos, tAv, eta, damping, N)
    GFZero = np.zeros((len(kVec), len(tRel)//2, len(tAv)), dtype='complex')
    GFT = np.concatenate((GFZero, GFT), axis=1)
    _, GFW = FT.FTOneOfTwoTimes(tRel, GFT)

    return GFW


def gfCohWLesser(kVec, wRel, tAv, eta, damping, N):
    tRel = FT.tVecFromWVec(wRel)
    tRelNeg = tRel[: len(tRel) // 2 + 1]
    GFT = gfCohExpDampingLesser(kVec, tRelNeg, tAv, eta, damping, N)
    GFZero = np.zeros((len(kVec), len(tRel) // 2 - 1, len(tAv)), dtype='complex')
    GFT = np.concatenate((GFT, GFZero), axis=1)
    _, GFW = FT.FTOneOfTwoTimes(tRel, GFT)

    return GFW



def gfPointTGS(kVec, tRel, tAv, eta):
    phState = getPhGS(eta)
    H = getH(eta)

    gk = - 2. * prms.t * np.sin(kVec)
    eK = 2. * prms.t * np.cos(kVec)
    x = eta * (np.diag(np.sqrt(np.arange((prms.maxPhotonNumber - 1)) + 1), -1) + np.diag(
        np.sqrt(np.arange((prms.maxPhotonNumber - 1)) + 1), +1))
    sinX = sciLin.sinm(x)
    cosX = sciLin.cosm(x)

    iHTRelMTAv, iHTRelPTAv, iHtSinCosTRel = setUpMatricies(H, cosX, eK, gk, sinX, tAv, tRel)

    expiHt, expiHtDash, exptRel = setUpExponentialMatricies(iHTRelMTAv, iHTRelPTAv, iHtSinCosTRel, kVec, tAv, tRel)

    GF = evaluateBraKet(expiHt, expiHtDash, exptRel, kVec, phState, tAv, tRel)

    return - 1j * GF

def gfGSExpDampingGreater(kVec, tRel, tAv, eta, damping):
    gsOcc = np.zeros(prms.chainLength, dtype=complex)
    gsOcc[prms.numberElectrons // 2 : - prms.numberElectrons // 2] = 1.
    print(gsOcc)
    dampExptRel = np.exp(- damping * np.abs(tRel))
    GF = gfPointTGS(kVec, tRel, tAv, eta)
    GFDamp = GF[:, :, :] * dampExptRel[None, :, None]
    GFOcc = GFDamp[:, :, :] * gsOcc[:, None, None]
    return GFOcc

def gfGSExpDampingLesser(kVec, tRel, tAv, eta, damping):
    gsOcc = np.zeros(prms.chainLength, dtype='double')
    gsOcc[ : prms.numberElectrons // 2 + 1] = 1.
    gsOcc[-prms.numberElectrons // 2 + 1 : ] = 1.
    print(gsOcc)
    dampExptRel = np.exp(- damping * np.abs(tRel))
    GF = gfPointTGS(kVec, tRel, tAv, eta)
    GFDamp = GF[:, :, :] * dampExptRel[None, :, None]
    GFOcc = GFDamp[:, :, :] * gsOcc[:, None, None]
    return GFOcc

def gfGSWGreater(kVec, wRel, tAv, eta, damping):
    tRel = FT.tVecFromWVec(wRel)
    tRelPos = tRel[len(tRel) // 2 : ]
    GFT = gfGSExpDampingGreater(kVec, tRelPos, tAv, eta, damping)
    GFZero = np.zeros((len(kVec), len(tRel)//2, len(tAv)), dtype='complex')
    GFT = np.concatenate((GFZero, GFT), axis=1)
    _, GFW = FT.FTOneOfTwoTimes(tRel, GFT)

    return GFW


def gfGSWLesser(kVec, wRel, tAv, eta, damping):
    tRel = FT.tVecFromWVec(wRel)
    tRelNeg = tRel[: len(tRel) // 2 + 1]
    GFT = gfGSExpDampingLesser(kVec, tRelNeg, tAv, eta, damping)
    GFZero = np.zeros((len(kVec), len(tRel) // 2 - 1, len(tAv)), dtype='complex')
    GFT = np.concatenate((GFT, GFZero), axis=1)
    _, GFW = FT.FTOneOfTwoTimes(tRel, GFT)

    return GFW




def evaluateBraKet(expiHt, expiHtDash, exptRel, kVec, phState, tAv, tRel):
    prod1 = np.zeros((len(tRel), len(tAv), prms.maxPhotonNumber), dtype='complex')
    for tAvInd in range(len(tAv)):
        for tRelInd in range(len(tRel)):
            prod1[tRelInd, tAvInd, :] = np.dot(expiHtDash[tRelInd, tAvInd, :, :], phState[:])
    prod2 = np.zeros((len(kVec), len(tRel), len(tAv), prms.maxPhotonNumber), dtype='complex')
    for tAvInd in range(len(tAv)):
        for tRelInd in range(len(tRel)):
            for kInd in range(len(kVec)):
                prod2[kInd, tRelInd, tAvInd, :] = np.dot(exptRel[kInd, tRelInd, :, :], prod1[tRelInd, tAvInd, :])
    prod3 = np.zeros((len(kVec), len(tRel), len(tAv), prms.maxPhotonNumber), dtype='complex')
    for tAvInd in range(len(tAv)):
        for tRelInd in range(len(tRel)):
            for kInd in range(len(kVec)):
                prod3[kInd, tRelInd, tAvInd, :] = np.dot(expiHt[tRelInd, tAvInd, :, :], prod2[kInd, tRelInd, tAvInd, :])
    GF = np.zeros((len(kVec), len(tRel), len(tAv)), dtype='complex')
    for tAvInd in range(len(tAv)):
        for tRelInd in range(len(tRel)):
            for kInd in range(len(kVec)):
                GF[kInd, tRelInd, tAvInd] = np.dot(np.conj(phState[:]), prod3[kInd, tRelInd, tAvInd, :])
    return GF


def setUpExponentialMatricies(iHTRelMTAv, iHTRelPTAv, iHtSinCosTRel, kVec, tAv, tRel):
    exptRel = np.zeros((len(kVec), len(tRel), prms.maxPhotonNumber, prms.maxPhotonNumber), dtype=complex)
    for tRelInd in range(len(tRel)):
        for kInd in range(len(kVec)):
            exptRel[kInd, tRelInd] = sciLin.expm(iHtSinCosTRel[kInd, tRelInd, :, :])
    expiHt = np.zeros((len(tRel), len(tAv), prms.maxPhotonNumber, prms.maxPhotonNumber), dtype=complex)
    expiHtDash = np.zeros((len(tRel), len(tAv), prms.maxPhotonNumber, prms.maxPhotonNumber), dtype=complex)
    for tAvInd in range(len(tAv)):
        for tRelInd in range(len(tRel)):
            expiHt[tRelInd, tAvInd, :, :] = sciLin.expm(iHTRelPTAv[tRelInd, tAvInd, :, :])
            expiHtDash[tRelInd, tAvInd, :, :] = sciLin.expm(iHTRelMTAv[tRelInd, tAvInd, :, :])
    return expiHt, expiHtDash, exptRel


def setUpMatricies(H, cosX, eK, gk, sinX, tAv, tRel):
    iHtSinCosTRel = - 1j * H[None, None, :, :] * tRel[None, :, None, None] \
                    - 1j * gk[:, None, None, None] * sinX[None, None, :, :] * tRel[None, :, None, None] \
                    - 1j * eK[:, None, None, None] * cosX[None, None, :, :] * tRel[None, :, None, None]
    iHTRelMTAv = - 1j * H[None, None, :, :] * (1. * tAv[None, :, None, None] - .5 * tRel[:, None, None, None])
    iHTRelPTAv = 1j * H[None, None, :, :] * (1. * tAv[None, :, None, None] + .5 * tRel[:, None, None, None])
    return iHTRelMTAv, iHTRelPTAv, iHtSinCosTRel


def getPhGS(eta):
    gs = arbOrder.findGS(eta, 3)
    gsJ = eF.J(gs)
    gsT = eF.T(gs)
    return phState.findPhotonGS([gsT, gsJ], eta, 3)


def getH(eta):
    gs = arbOrder.findGS(eta, 3)
    gsJ = eF.J(gs)
    gsT = eF.T(gs)
    return numH.setupPhotonHamiltonianInf(gsT, gsJ, eta)

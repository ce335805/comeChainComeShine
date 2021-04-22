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


def gfPointTCoh(kPoint, tRel, tAv, eta, N):
    phState = coherentState.getCoherentStateForN(N)
    H = getH(eta)

    timeStart = time.time()
    iHTRelMTAv, iHTRelPTAv, iHtSinCosTRel = setUpMatricies(kPoint, H, tAv, tRel, eta)
    timeStop = time.time()
    print("setUpMatricies take: {}".format(timeStop - timeStart))

    timeStart = time.time()
    expiHt, expiHtDash, exptRel = setUpExponentialMatricies(iHTRelMTAv, iHTRelPTAv, iHtSinCosTRel, tAv, tRel)
    timeStop = time.time()
    print("setUpExponentialMatricies take: {}".format(timeStop - timeStart))

    timeStart = time.time()
    GF = evaluateBraKet(expiHt, expiHtDash, exptRel, kPoint, phState, tAv, tRel)
    timeStop = time.time()
    print("evaluateBraKet take: {}".format(timeStop - timeStart))

    return - 1j * GF


def gfCohExpDamping(kPoint, tRel, tAv, eta, damping, N):
    dampExptRel = np.exp(- damping * np.abs(tRel))
    GF = gfPointTCoh(kPoint, tRel, tAv, eta, N)
    GFDamp = GF[:, :] * dampExptRel[:, None]
    return GFDamp


def gfCohWGreater(kPoint, wRel, tAv, eta, damping, N):
    tRel = FT.tVecFromWVec(wRel)
    tRelPos = tRel[len(tRel) // 2:]
    GFT = gfCohExpDamping(kPoint, tRelPos, tAv, eta, damping, N)
    GFZero = np.zeros((len(tRel) // 2, len(tAv)), dtype='complex')
    GFT = np.concatenate((GFZero, GFT), axis=0)
    _, GFW = FT.FTOneOfTwoTimesPoint(tRel, GFT)

    return GFW


def gfCohWLesser(kPoint, wRel, tAv, eta, damping, N):
    tRel = FT.tVecFromWVec(wRel)
    tRelNeg = tRel[: len(tRel) // 2 + 1]
    GFT = gfCohExpDamping(kPoint, tRelNeg, tAv, eta, damping, N)
    GFZero = np.zeros((len(tRel) // 2 - 1, len(tAv)), dtype='complex')
    GFT = np.concatenate((GFT, GFZero), axis=0)
    _, GFW = FT.FTOneOfTwoTimesPoint(tRel, GFT)

    return GFW


def gfPointTGS(kPoint, tRel, tAv, eta):
    phState = getPhGS(eta)
    H = getH(eta)

    iHTRelMTAv, iHTRelPTAv, iHtSinCosTRel = setUpMatricies(kPoint, H, tAv, tRel, eta)

    expiHt, expiHtDash, exptRel = setUpExponentialMatricies(iHTRelMTAv, iHTRelPTAv, iHtSinCosTRel, tAv, tRel)

    GF = evaluateBraKet(expiHt, expiHtDash, exptRel, kPoint, phState, tAv, tRel)

    return - 1j * GF



def gfGSExpDamping(kPoint, tRel, tAv, eta, damping):
    dampExptRel = np.exp(- damping * np.abs(tRel))
    GF = gfPointTGS(kPoint, tRel, tAv, eta)
    GFDamp = GF[:, :] * dampExptRel[:, None]
    return GFDamp


def gfGSWGreater(kPoint, wRel, tAv, eta, damping):
    tRel = FT.tVecFromWVec(wRel)
    tRelPos = tRel[len(tRel) // 2:]
    GFT = gfGSExpDamping(kPoint, tRelPos, tAv, eta, damping)
    GFZero = np.zeros((len(tRel) // 2, len(tAv)), dtype='complex')
    GFT = np.concatenate((GFZero, GFT), axis=0)
    _, GFW = FT.FTOneOfTwoTimesPoint(tRel, GFT)

    return GFW


def gfGSWLesser(kPoint, wRel, tAv, eta, damping):
    tRel = FT.tVecFromWVec(wRel)
    tRelNeg = tRel[: len(tRel) // 2 + 1]
    GFT = gfGSExpDamping(kPoint, tRelNeg, tAv, eta, damping)
    GFZero = np.zeros((len(tRel) // 2 - 1, len(tAv)), dtype='complex')
    GFT = np.concatenate((GFT, GFZero), axis=0)
    _, GFW = FT.FTOneOfTwoTimesPoint(tRel, GFT)

    return GFW


def evaluateBraKet(expiHt, expiHtDash, exptRel, kVec, phState, tAv, tRel):
    prod1 = np.zeros((len(tRel), len(tAv), prms.maxPhotonNumber), dtype='complex')
    for tAvInd in range(len(tAv)):
        for tRelInd in range(len(tRel)):
            prod1[tRelInd, tAvInd, :] = np.dot(expiHtDash[tRelInd, tAvInd, :, :], phState[:])
    prod2 = np.zeros((len(tRel), len(tAv), prms.maxPhotonNumber), dtype='complex')
    for tAvInd in range(len(tAv)):
        for tRelInd in range(len(tRel)):
                prod2[tRelInd, tAvInd, :] = np.dot(exptRel[tRelInd, :, :], prod1[tRelInd, tAvInd, :])
    prod3 = np.zeros((len(tRel), len(tAv), prms.maxPhotonNumber), dtype='complex')
    for tAvInd in range(len(tAv)):
        for tRelInd in range(len(tRel)):
                prod3[tRelInd, tAvInd, :] = np.dot(expiHt[tRelInd, tAvInd, :, :], prod2[tRelInd, tAvInd, :])
    GF = np.zeros((len(tRel), len(tAv)), dtype='complex')
    for tAvInd in range(len(tAv)):
        for tRelInd in range(len(tRel)):
                GF[tRelInd, tAvInd] = np.dot(np.conj(phState[:]), prod3[tRelInd, tAvInd, :])
    return GF


def setUpExponentialMatricies(iHTRelMTAv, iHTRelPTAv, iHtSinCosTRel, tAv, tRel):
    print("iHtSinCosTRel.shape = {}".format(iHtSinCosTRel.shape))
    exptRel = np.zeros((len(tRel), prms.maxPhotonNumber, prms.maxPhotonNumber), dtype=complex)
    for tRelInd in range(len(tRel)):
        exptRel[tRelInd] = sciLin.expm(iHtSinCosTRel[tRelInd, :, :])
    expiHt = np.zeros((len(tRel), len(tAv), prms.maxPhotonNumber, prms.maxPhotonNumber), dtype=complex)
    expiHtDash = np.zeros((len(tRel), len(tAv), prms.maxPhotonNumber, prms.maxPhotonNumber), dtype=complex)
    for tAvInd in range(len(tAv)):
        for tRelInd in range(len(tRel)):
            expiHt[tRelInd, tAvInd, :, :] = sciLin.expm(iHTRelPTAv[tRelInd, tAvInd, :, :])
            expiHtDash[tRelInd, tAvInd, :, :] = sciLin.expm(iHTRelMTAv[tRelInd, tAvInd, :, :])
    return expiHt, expiHtDash, exptRel


def setUpMatricies(kPoint, H, tAv, tRel, eta):
    gk = - 2. * prms.t * np.sin(kPoint)
    eK = 2. * prms.t * np.cos(kPoint)
    x = eta * (np.diag(np.sqrt(np.arange((prms.maxPhotonNumber - 1)) + 1), -1) + np.diag(
        np.sqrt(np.arange((prms.maxPhotonNumber - 1)) + 1), +1))
    sinX = sciLin.sinm(x)
    cosX = sciLin.cosm(x)
    iHtSinCosTRel = - 1j * H[None, :, :] * tRel[:, None, None] \
                    - 1j * gk * sinX[None, :, :] * tRel[:, None, None] \
                    - 1j * eK * cosX[None, :, :] * tRel[:, None, None]
    iHTRelMTAv = - 1j * H[None, None, :, :] * (1. * tAv[None, :, None, None] - .5 * tRel[:, None, None, None])
    iHTRelPTAv = 1j * H[None, None, :, :] * (1. * tAv[None, :, None, None] + .5 * tRel[:, None, None, None])
    return iHTRelMTAv, iHTRelPTAv, iHtSinCosTRel


def getPhGS(eta):
    gsJ = 0.
    gsE = np.zeros(prms.chainLength)
    gsE[0: prms.numberElectrons // 2 + 1] = 1.
    gsE[- prms.numberElectrons // 2 + 1:] = 1.
    kVec = np.linspace(0, 2. * np.pi, prms.chainLength, endpoint=False)
    cosK = np.cos(kVec)
    gsT = 2. * prms.t * np.sum(np.multiply(cosK, gsE))

    return phState.findPhotonGS([gsT, gsJ], eta, 3)


def getH(eta):
    gsJ = 0.
    gsE = np.zeros(prms.chainLength)
    gsE[0: prms.numberElectrons // 2 + 1] = 1.
    gsE[- prms.numberElectrons // 2 + 1:] = 1.
    kVec = np.linspace(0, 2. * np.pi, prms.chainLength, endpoint=False)
    cosK = np.cos(kVec)
    gsT = 2. * prms.t * np.sum(np.multiply(cosK, gsE))

    return numH.setupPhotonHamiltonianInf(gsT, gsJ, eta)

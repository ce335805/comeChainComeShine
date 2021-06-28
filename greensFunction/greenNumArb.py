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

def gfNumPointTGreater(kVec, tVec, eta):

    phGS = getPhGS(eta)

    H = getH(eta)

    gk = - 2. * prms.t * np.sin(kVec)
    eK = 2. * prms.t * np.cos(kVec)
    x = eta * (np.diag(np.sqrt(np.arange((prms.maxPhotonNumber - 1)) + 1), -1) + np.diag(
        np.sqrt(np.arange((prms.maxPhotonNumber - 1)) + 1), +1))
    sinX = sciLin.sinm(x)
    cosX = sciLin.cosm(x)
    #iHt = 1j * H[None, :, :] * tVec[:, None, None]
    #iHtSinCos = - 1j * H[None, None, :, :] * tVec[None, :, None, None] \
    #            - 1j * gk[:, None, None, None] * sinX[None, None, :, :] * tVec[None, :, None, None] \
    #            - 1j * eK[:, None, None, None] * cosX[None, None, :, :] * tVec[None, :, None, None]

    hEvals, hEvecs = np.linalg.eigh(H)
    HtSinCosK = H[None, :, :] \
                + gk[:, None, None] * sinX[None, :, :] \
                + eK[:, None, None] * cosX[None, :, :]
    eValsK, eVecsK = np.linalg.eigh(HtSinCosK)

    GF = np.zeros((len(kVec), len(tVec)), dtype='complex')
    for tInd, t in enumerate(tVec):
        expiHt = np.dot(hEvecs, np.dot(np.diag(np.exp(1j * hEvals * t)), np.transpose(np.conj(hEvecs))))
        for kInd, k in enumerate(kVec):
            expiHSinCost = np.dot(eVecsK[kInd, :, :], np.dot(np.diag(np.exp(-1j * eValsK[kInd, :,] * t)) , np.transpose(np.conj(eVecsK[kInd, :, :]))))
            prod1 = np.dot( expiHSinCost, phGS)
            prod2 = np.dot( expiHt, prod1)
            #prod1 = np.dot( sciLin.expm(iHtSinCos[kInd, tInd, :, :]), phGS)
            #prod2 = np.dot( sciLin.expm(iHt[tInd, :, :]), prod1)
            res = np.dot(np.conj(phGS), prod2)
            GF[kInd, tInd] = res

    return - 1j * GF


def gfNumVecTGreater(kVec, tVec, eta, damping):
    if(len(kVec) == prms.chainLength):
        gs = arbOrder.findGS(eta, 3)
    else:
        gs = np.ones(len(kVec), dtype=complex)
    _, occupations = np.meshgrid(np.ones(tVec.shape), gs[:])
    GF = gfNumPointTGreater(kVec, tVec, eta)
    GF = np.multiply(1 - occupations, GF)
    dampingArr, _ = np.meshgrid(np.exp(- damping * np.abs(tVec)), np.ones(kVec.shape))
    GF = np.multiply(dampingArr, GF)

    return GF

def gfNumPointTLesser(kVec, tVec, eta):

    phGS = getPhGS(eta)
    #phGS = coherentState.getCoherentStateForN(3.)
    H = getH(eta)

    gk = - 2. * prms.t * np.sin(kVec)
    eK = 2. * prms.t * np.cos(kVec)
    x = eta * (np.diag(np.sqrt(np.arange((prms.maxPhotonNumber - 1)) + 1), -1) + np.diag(
        np.sqrt(np.arange((prms.maxPhotonNumber - 1)) + 1), +1))
    sinX = sciLin.sinm(x)
    cosX = sciLin.cosm(x)
    #iHt = 1j * H[None, :, :] * tVec[:, None, None]
    #iHtSinCos = - 1j * H[None, None, :, :] * tVec[None, :, None, None] \
    #            - 1j * gk[:, None, None, None] * sinX[None, None, :, :] * tVec[None, :, None, None] \
    #            - 1j * eK[:, None, None, None] * cosX[None, None, :, :] * tVec[None, :, None, None]

    hEvals, hEvecs = np.linalg.eigh(H)
    HtSinCosK = H[None, :, :] \
                - gk[:, None, None] * sinX[None, :, :] \
                - eK[:, None, None] * cosX[None, :, :]
    eValsK, eVecsK = np.linalg.eigh(HtSinCosK)

    GF = np.zeros((len(kVec), len(tVec)), dtype='complex')
    for tInd, t in enumerate(tVec):
        expiHt = np.dot(hEvecs, np.dot(np.diag(np.exp(-1j * hEvals * t)), np.transpose(np.conj(hEvecs))))
        for kInd, k in enumerate(kVec):
            expiHSinCost = np.dot(eVecsK[kInd, :, :], np.dot(np.diag(np.exp(1j * eValsK[kInd, :,] * t)) , np.transpose(np.conj(eVecsK[kInd, :, :]))))
            prod1 = np.dot( expiHt, phGS)
            prod2 = np.dot( expiHSinCost, prod1)
            #prod1 = np.dot( sciLin.expm(iHtSinCos[kInd, tInd, :, :]), phGS)
            #prod2 = np.dot( sciLin.expm(iHt[tInd, :, :]), prod1)
            res = np.dot(np.conj(phGS), prod2)
            GF[kInd, tInd] = res

    return - 1j * GF


def gfNumVecTLesser(kVec, tVec, eta, damping):
    if(len(kVec) == prms.chainLength):
        gs = arbOrder.findGS(eta, 3)
    else:
        gs = np.ones(len(kVec), dtype=complex)
    _, occupations = np.meshgrid(np.ones(tVec.shape), gs[:])
    GF = gfNumPointTLesser(kVec, tVec, eta)
    GF = np.multiply(occupations, GF)

    dampingArr, _ = np.meshgrid(np.exp(- damping * np.abs(tVec)), np.ones(kVec.shape))
    GF = np.multiply(dampingArr, GF)

    return GF


def numGreenVecWGreater(kVec, wVec, eta, damping):
    tVec = FT.tVecFromWVec(wVec)
    tVecPos = tVec[len(tVec) // 2 : ]
    GFT = gfNumVecTGreater(kVec, tVecPos, eta, damping)
    GFZero = np.zeros((len(kVec), len(tVec)//2), dtype='complex')
    GFT = np.concatenate((GFZero, GFT), axis=1)

    wVecCheck, GFW = FT.FT(tVec, GFT)

    return GFW

def numGreenVecWLesser(kVec, wVec, eta, damping):
    tVec = FT.tVecFromWVec(wVec)
    tVecPos = tVec[len(tVec) // 2 : ]
    GFT = gfNumVecTLesser(kVec, tVecPos, eta, damping)
    GFZero = np.zeros((len(kVec), len(tVec) // 2), dtype='complex')
    GFT = np.concatenate((GFZero, GFT), axis=1)

    wVecCheck, GFW = FT.FT(tVec, GFT)

    return GFW

def spectralGreater(kVec, wVec, eta, damping):
    return -2. * np.imag(numGreenVecWGreater(kVec, wVec, eta, damping))

def spectralLesser(kVec, wVec, eta, damping):
    return -2. * np.imag(numGreenVecWLesser(kVec, wVec, eta, damping))


def getPhGS(eta):
    gs = arbOrder.findGS(eta, 3)
    gsJ = eF.J(gs)
    gsT = eF.T(gs)
    return phState.findPhotonGS([gsT, gsJ], eta, 2)


def getH(eta):
    gs = arbOrder.findGS(eta, 3)
    gsJ = eF.J(gs)
    gsT = eF.T(gs)
    return numH.setupPhotonHamiltonianInf(gsT, gsJ, eta)


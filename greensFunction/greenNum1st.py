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

    phGS = getPhGSH1(eta)
    H = getH1(eta)

    x = np.diag(np.sqrt(np.arange((prms.maxPhotonNumber - 1)) + 1), -1) + np.diag(
        np.sqrt(np.arange((prms.maxPhotonNumber - 1)) + 1), +1)
    gk = - 2. * prms.t * np.sin(kVec) * eta
    eK = 2. * prms.t * np.cos(kVec)
    photonOne = np.diag(np.ones(prms.maxPhotonNumber))

    hEvals, hEvecs = np.linalg.eigh(H)
    HtSinCosK = H[None, :, :] \
                + gk[:, None, None] * x[None, :, :] \
                + eK[:, None, None] * photonOne[None, :, :]
    eValsK, eVecsK = np.linalg.eigh(HtSinCosK)

    GF = np.zeros((len(kVec), len(tVec)), dtype='complex')
    for tInd, t in enumerate(tVec):
        expiHt = np.dot(hEvecs, np.dot(np.diag(np.exp(1j * hEvals * t)), np.transpose(np.conj(hEvecs))))
        for kInd, k in enumerate(kVec):
            expiHSinCost = np.dot(eVecsK[kInd, :, :], np.dot(np.diag(np.exp(-1j * eValsK[kInd, :,] * t)) , np.transpose(np.conj(eVecsK[kInd, :, :]))))
            prod1 = np.dot( expiHSinCost, phGS)
            prod2 = np.dot( expiHt, prod1)
            res = np.dot(np.conj(phGS), prod2)
            GF[kInd, tInd] = res

    return - 1j * GF


def gfNumVecTGreater(kVec, tVec, eta, damping):
    gs = arbOrder.findGS(eta, 1)
    _, occupations = np.meshgrid(np.ones(tVec.shape), gs[:])
    GF = gfNumPointTGreater(kVec, tVec, eta)
    GF = np.multiply(1 - occupations, GF)
    dampingArr, _ = np.meshgrid(np.exp(- damping * np.abs(tVec)), np.ones(kVec.shape))
    GF = np.multiply(dampingArr, GF)

    return GF

def gfNumPointTLesser(kVec, tVec, eta):

    phGS = getPhGSH1(eta)
    H = getH1(eta)

    x = np.diag(np.sqrt(np.arange((prms.maxPhotonNumber - 1)) + 1), -1) + np.diag(
        np.sqrt(np.arange((prms.maxPhotonNumber - 1)) + 1), +1)
    gk = - 2. * prms.t * np.sin(kVec) * eta
    eK = 2. * prms.t * np.cos(kVec)
    photonOne = np.diag(np.ones(prms.maxPhotonNumber))

    hEvals, hEvecs = np.linalg.eigh(H)
    HtSinCosK = H[None, :, :] - \
                gk[:, None, None] * x[None, :, :] - \
                eK[:, None, None] * photonOne[None, :, :]
    eValsK, eVecsK = np.linalg.eigh(HtSinCosK)

    GF = np.zeros((len(kVec), len(tVec)), dtype='complex')
    for tInd, t in enumerate(tVec):
        expiHt = np.dot(hEvecs, np.dot(np.diag(np.exp(-1j * hEvals * t)), np.transpose(np.conj(hEvecs))))
        for kInd, k in enumerate(kVec):
            expiHSinCost = np.dot(eVecsK[kInd, :, :], np.dot(np.diag(np.exp(1j * eValsK[kInd, :,] * t)) , np.transpose(np.conj(eVecsK[kInd, :, :]))))
            prod1 = np.dot( expiHt, phGS)
            prod2 = np.dot( expiHSinCost, prod1)
            res = np.dot(np.conj(phGS), prod2)
            GF[kInd, tInd] = res

    return - 1j * GF

def gfNumVecTLesser(kVec, tVec, eta, damping):
    gs = arbOrder.findGS(eta, 1)
    _, occupations = np.meshgrid(np.ones(tVec.shape), gs[:])
    GF = gfNumPointTLesser(kVec, tVec, eta)
    GF = np.multiply(occupations, GF)

    dampingArr, _ = np.meshgrid(np.exp(- damping * np.abs(tVec)), np.ones(kVec.shape))
    GF = np.multiply(dampingArr, GF)

    return GF


def numGreenVecWGreater(kVec, wVec, eta, damping):
    tVec = FT.tVecFromWVec(wVec)
    tVecPos = tVec[len(tVec) // 2: ]
    GFT = gfNumVecTGreater(kVec, tVecPos, eta, damping)
    GFZero = np.zeros((len(kVec), len(tVec)//2), dtype='complex')
    GFT = np.concatenate((GFZero, GFT), axis=1)

    wVecCheck, GFW = FT.FT(tVec, GFT)

    return GFW

def numGreenVecWLesser(kVec, wVec, eta, damping):
    tVec = FT.tVecFromWVec(wVec)
    tVecPos = tVec[len(tVec) // 2: ]
    GFT = gfNumVecTLesser(kVec, tVecPos, eta, damping)
    GFZero = np.zeros((len(kVec), len(tVec)//2), dtype='complex')
    GFT = np.concatenate((GFZero, GFT), axis=1)

    wVecCheck, GFW = FT.FT(tVec, GFT)

    return GFW


def spectralGreater(kVec, wVec, eta, damping):
    return -2. * np.imag(numGreenVecWGreater(kVec, wVec, eta, damping))

def spectralLesser(kVec, wVec, eta, damping):
    return -2. * np.imag(numGreenVecWLesser(kVec, wVec, eta, damping))

def getPhGSH1(eta):
    gs = arbOrder.findGS(eta, 1)
    gsJ = eF.J(gs)
    gsT = eF.T(gs)
    return phState.findPhotonGS([gsT, gsJ], eta, 1)


def getH1(eta):
    gs = arbOrder.findGS(eta, 1)
    gsJ = eF.J(gs)
    gsT = eF.T(gs)
    return numH.setupPhotonHamiltonian1st(gsT, gsJ, eta)





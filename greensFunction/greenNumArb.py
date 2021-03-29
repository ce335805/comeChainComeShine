import numpy as np
import sec_order.analyticalEGS as anaGS
import globalSystemParams as prms
import energyFunctions as eF
from arb_order import photonState as phState
from arb_order import numHamiltonians as numH
import scipy.linalg as sciLin
import fourierTrafo as FT
from arb_order import arbOrder


def gfNumPointTGreater(kVec, tVec, eta):

    phGS = getPhGS(eta)
    H = getH(eta)

    gk = - 2. * prms.t * np.sin(kVec)
    eK = 2. * prms.t * np.cos(kVec)
    x = eta * (np.diag(np.sqrt(np.arange((prms.maxPhotonNumber - 1)) + 1), -1) + np.diag(
        np.sqrt(np.arange((prms.maxPhotonNumber - 1)) + 1), +1))
    sinX = sciLin.sinm(x)
    cosX = sciLin.cosm(x)
    iHt = 1j * H[None, :, :] * tVec[:, None, None]
    iHtSinCos = - 1j * H[None, None, :, :] * tVec[None, :, None, None] \
                - 1j * gk[:, None, None, None] * sinX[None, None, :, :] * tVec[None, :, None, None] \
                - 1j * eK[:, None, None, None] * cosX[None, None, :, :] * tVec[None, :, None, None]


    GF = np.zeros((len(kVec), len(tVec)), dtype='complex')
    for tInd in range(len(tVec)):
        for kInd in range(len(kVec)):
            prod1 = np.dot( sciLin.expm(iHtSinCos[kInd, tInd, :, :]), phGS)
            prod2 = np.dot( sciLin.expm(iHt[tInd, :, :]), prod1)
            res = np.dot(np.conj(phGS), prod2)
            GF[kInd, tInd] = res

    return - 1j * GF


def gfNumVecTGreater(kVec, tVec, eta, damping):
    initialState = np.zeros(prms.chainLength, dtype='double')
    initialState[0: prms.numberElectrons] = 1.0
    gs = arbOrder.findGS(initialState, eta, 1)
    _, occupations = np.meshgrid(np.ones(tVec.shape), gs[:])
    GF = gfNumPointTGreater(kVec, tVec, eta)
    GF = np.multiply(1 - occupations, GF)
    dampingArr, _ = np.meshgrid(np.exp(- damping * np.abs(tVec)), np.ones(kVec.shape))
    GF = np.multiply(dampingArr, GF)

    return GF

def gfNumPointTLesser(kVec, tVec, eta):

    phGS = getPhGS(eta)

    H = getH(eta)
    gk = - 2. * prms.t * np.sin(kVec)
    eK = 2. * prms.t * np.cos(kVec)
    x = eta * (np.diag(np.sqrt(np.arange((prms.maxPhotonNumber - 1)) + 1), -1) + np.diag(
        np.sqrt(np.arange((prms.maxPhotonNumber - 1)) + 1), +1))
    sinX = sciLin.sinm(x)
    cosX = sciLin.cosm(x)
    iHt = 1j * H[None, :, :] * tVec[:, None, None]
    iHtSinCos = - 1j * H[None, None, :, :] * tVec[None, :, None, None] \
                + 1j * gk[:, None, None, None] * sinX[None, None, :, :] * tVec[None, :, None, None] \
                + 1j * eK[:, None, None, None] * cosX[None, None, :, :] * tVec[None, :, None, None]

    GF = np.zeros((len(kVec), len(tVec)), dtype='complex')
    for tInd in range(len(tVec)):
        prod1 = np.dot(phGS, sciLin.expm(iHt[tInd, :, :]))
        for kInd in range(len(kVec)):
            prod2 = np.dot(sciLin.expm(iHtSinCos[kInd, tInd, :, :]), phGS)
            GF[kInd, tInd] = np.dot(prod1, prod2)

    return 1j * GF


def gfNumVecTLesser(kVec, tVec, eta, damping):
    initialState = np.zeros(prms.chainLength, dtype='double')
    initialState[0: prms.numberElectrons] = 1.0
    gs = arbOrder.findGS(initialState, eta, 1)
    _, occupations = np.meshgrid(np.ones(tVec.shape), gs[:])
    GF = gfNumPointTLesser(kVec, tVec, eta)
    GF = np.multiply(occupations, GF)

    dampingArr, _ = np.meshgrid(np.exp(- damping * np.abs(tVec)), np.ones(kVec.shape))
    GF = np.multiply(dampingArr, GF)

    return GF


def numGreenVecWGreater(kVec, wVec, eta, damping):
    tVec = FT.tVecFromWVec(wVec)
    tVecPos = tVec[len(tVec)//2 + 1: ]
    GFT = gfNumVecTGreater(kVec, tVecPos, eta, damping)
    GFZero = np.zeros((len(kVec), len(tVec)//2 + 1), dtype='complex')
    GFT = np.concatenate((GFZero, GFT), axis=1)

    wVecCheck, GFW = FT.FT(tVec, GFT)

    return GFW

def spectralGreater(kVec, wVec, eta, damping):
    return -2. * np.imag(numGreenVecWGreater(kVec, wVec, eta, damping))


def numGreenVecWLesser(kVec, wVec, eta, damping):
    tVec = FT.tVecFromWVec(wVec)
    tVecNeg = tVec[: len(tVec) // 2 + 1]
    GFT = gfNumVecTLesser(kVec, tVecNeg, eta, damping)
    GFZero = np.zeros((len(kVec), len(tVec) // 2), dtype='complex')
    GFT = np.concatenate((GFT, GFZero), axis=1)
    wVecCheck, GFW = FT.FT(tVec, GFT)
    assert ((np.abs(wVec - wVecCheck) < 1e-10).all)
    return GFW



def getPhGS(eta):
    initialState = np.zeros(prms.chainLength, dtype='double')
    initialState[0: prms.numberElectrons] = 1.0
    gs = arbOrder.findGS(initialState, eta, 3)
    gsJ = eF.J(gs)
    gsT = eF.T(gs)
    return phState.findPhotonGS([gsT, gsJ], eta, 3)


def getH(eta):
    initialState = np.zeros(prms.chainLength, dtype='double')
    initialState[0: prms.numberElectrons] = 1.0
    gs = arbOrder.findGS(initialState, eta, 3)
    gsJ = eF.J(gs)
    gsT = eF.T(gs)
    return numH.setupPhotonHamiltonianInf(gsT, gsJ, eta)


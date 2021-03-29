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

    phGS = getPhGSH1(eta)
    print(phGS)
    H = getH1(eta)

    x = np.diag(np.sqrt(np.arange((prms.maxPhotonNumber - 1)) + 1), -1) + np.diag(
        np.sqrt(np.arange((prms.maxPhotonNumber - 1)) + 1), +1)
    gk = - 2. * prms.t * np.sin(kVec) * eta
    eK = 2. * prms.t * np.cos(kVec)
    photonOne = np.diag(np.ones(prms.maxPhotonNumber))
    iHt = 1j * H[None, :, :] * tVec[:, None, None]
    iHtSinCos = - 1j * H[None, None, :, :] * tVec[None, :, None, None] \
                - 1j * gk[:, None, None, None] * x[None, None, :, :] * tVec[None, :, None, None] \
                - 1j * eK[:, None, None, None] * photonOne[None, None, :, :] * tVec[None, :, None, None]


    GF = np.zeros((len(kVec), len(tVec)), dtype='complex')
    for tInd in range(len(tVec)):
        for kInd in range(len(kVec)):
            prod1 = np.dot( sciLin.expm(iHtSinCos[kInd, tInd, :, :]), phGS)
            prod2 = np.dot( sciLin.expm(iHt[tInd, :, :]), prod1)
            res = np.dot(np.conj(phGS), prod2)
            GF[kInd, tInd] = res

    return - 1j * GF

def gfNumPointTGreaterTrafo(kVec, tVec, eta):

    phGS = getPhGSH1(0.)
    H = getDiagPtH()

    p = np.diag(np.sqrt(np.arange((prms.maxPhotonNumber - 1)) + 1), -1) - np.diag(
        np.sqrt(np.arange((prms.maxPhotonNumber - 1)) + 1), +1)
    gk = - 2. * prms.t * np.sin(kVec) * eta
    iHt = 1j * H[None, :, :] * tVec[:, None, None]
    cohExponent = (gk[:, None, None] / prms.w0) * p[None, :, :]
    mCohExponent = - (gk[:, None, None] / prms.w0) * p[None, :, :]

    GF = np.zeros((len(kVec), len(tVec)), dtype='complex')
    for tInd in range(len(tVec)):
        for kInd in range(len(kVec)):
            expIHT = sciLin.expm(iHt[tInd, :, :])
            expCoh = sciLin.expm(cohExponent[kInd, :, :])
            expMCoh = sciLin.expm(mCohExponent[kInd, :, :])
            prod1 = np.dot(expCoh, phGS)
            prod2 = np.dot(np.conj(expIHT), prod1)
            prod3 = np.dot(expMCoh, prod2)
            prod4 = np.dot(expIHT, prod3)

            GF[kInd, tInd] = np.dot(np.conj(phGS), prod4)

    gsJ = getGsJ(eta)
    epsK = 2. * prms.t * np.cos(kVec)
    coupling = - eta ** 2 / prms.w0 * \
               (2. * ( gsJ * (-2.) * prms.t * np.sin(kVec)) + (-2. * prms.t * np.sin(kVec))**2)


    iPhiT = -1j * (epsK[:, None] + coupling[:, None]) * tVec[None, :]
    GF = np.multiply(np.exp(iPhiT), GF)

    return - 1j * GF

def gfNumVecTGreater(kVec, tVec, eta, damping):
    GF = gfNumPointTGreater(kVec, tVec, eta)
    #GF = gfNumPointTGreaterTrafo(kVec, tVec, eta)
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
    iHt = 1j * H[None, :, :] * tVec[:, None, None]
    iHtSinCos = - iHt \
                + 1j * gk[:, None, None, None] * x[None, None, :, :] * tVec[None, :, None, None] \
                + 1j * eK[:, None, None, None] * tVec[None, :, None, None] * photonOne[None, None, :, :]

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



def getPhGSH1(eta):
    initialState = np.zeros(prms.chainLength, dtype='double')
    initialState[0: prms.numberElectrons] = 1.0
    gs = arbOrder.findGS(initialState, eta, 1)
    gsJ = eF.J(gs)
    gsT = eF.T(gs)
    return phState.findPhotonGS([gsT, gsJ], eta, 1)


def getH1(eta):
    initialState = np.zeros(prms.chainLength, dtype='double')
    initialState[0: prms.numberElectrons] = 1.0
    gs = arbOrder.findGS(initialState, eta, 1)
    gsJ = eF.J(gs)
    gsT = eF.T(gs)
    return numH.setupPhotonHamiltonian1st(gsT, gsJ, eta)

def getDiagPtH():
    indices = np.arange(prms.maxPhotonNumber)
    diagonal = prms.w0 * (indices + .5)
    return np.diag(diagonal, 0)


def getGsJ(eta):
    initialState = np.zeros(prms.chainLength, dtype='double')
    initialState[0: prms.numberElectrons] = 1.0
    gs = arbOrder.findGS(initialState, eta, 1)
    gsJ = eF.J(gs)
    return gsJ





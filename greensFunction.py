import numpy as np
import sec_order.analyticalEGS as anaGS
import globalSystemParams as prms
import energyFunctions as eF
from arb_order import photonState as phState
from arb_order import numHamiltonians as numH
import scipy.linalg as sciLin
import time


def g0T(kPoint, tPoint):
    return - 1j * np.exp(-1j * 2. * prms.t * np.cos(kPoint[:, None]) * tPoint[None, :])


def g0VecT(kVec, tVec):
    initialState = np.zeros(prms.chainLength + 1, dtype='double')
    gs = anaGS.findGS1st(initialState, 0.)
    _, occupations = np.meshgrid(np.ones(tVec.shape), gs[0: -1])
    GF = g0T(kVec, tVec)
    GF = np.multiply(1 - occupations, GF)
    return GF


def anaGreenPointTGreater(kPoint, tPoint, gsJ, eta):
    epsK = 2. * prms.t * np.cos(kPoint[:, None])
    coupling = - eta ** 2 / prms.w0 * \
               (2. * (-2. * gsJ * prms.t * np.sin(kPoint[:, None])) + (-2. * prms.t * np.sin(kPoint[:, None]))**2)

    eTime = -1j * epsK * tPoint[None, :] - 1j * coupling * tPoint[None, :]
    ptTime = - (- 2. * eta * prms.t * np.sin(kPoint[:, None]))**2 / prms.w0**2 * (1. - np.exp(-1j * prms.w0 * tPoint[None, :]))
    return -1j * np.exp(eTime + ptTime)

def anaGreenPointTLesser(kPoint, tPoint, gsJ, eta):
    epsK = -2. * prms.t * np.cos(kPoint[:, None])
    coupling = + eta ** 2 / prms.w0 * \
               (2. * (-2. * gsJ * prms.t * np.sin(kPoint[:, None])) - (-2. * prms.t * np.sin(kPoint[:, None]))**2)

    eTime = - 1j * epsK * tPoint[None, :] - 1j * coupling * tPoint[None, :]
    ptTime = - (- 2. * eta * prms.t * np.sin(kPoint[:, None]))**2 / prms.w0**2 * (1. - np.exp(-1j * prms.w0 * tPoint[None, :]))
    return 1j * np.exp(eTime + ptTime)

def anaGreenVecTGreater(kVec, tVec, eta):
    initialState = np.zeros(prms.chainLength + 1, dtype='double')
    gs = anaGS.findGS1st(initialState, eta)
    gsJ = eF.J(gs[0: -1])
    _, occupations = np.meshgrid(np.ones(tVec.shape), gs[0: -1])
    GF = anaGreenPointTGreater(kVec, tVec, gsJ, eta)
    GF = np.multiply(1 - occupations, GF)

    return GF

def anaGreenVecTLesser(kVec, tVec, eta):
    initialState = np.zeros(prms.chainLength + 1, dtype='double')
    gs = anaGS.findGS1st(initialState, eta)
    gsJ = eF.J(gs[0: -1])
    _, occupations = np.meshgrid(np.ones(tVec.shape), gs[0: -1])
    GF = anaGreenPointTLesser(kVec, tVec, gsJ, eta)
    GF = np.multiply(occupations, GF)

    return GF

def anaGreenVecTComplete(kVec, tVec, eta, damping):
    tLess = tVec[:len(tVec)//2 + 1]
    tGreat = tVec[len(tVec)//2 + 1:]
    GFL = anaGreenVecTLesser(kVec, tLess, eta)
    GFG = anaGreenVecTGreater(kVec, tGreat, eta)
    GF = np.concatenate((GFL, GFG), axis=1)

    dampingArr, _ = np.meshgrid(np.exp(- damping * np.abs(tVec)), np.ones(kVec.shape))
    GF = np.multiply(dampingArr, GF)

    print("GF.shape = {}".format(GF.shape))

    return GF


def gfNumPointT(kVec, tVec, eta):
    print("calculating GF numrically")

    phGS = getPhGSH1(eta)

    H = getH1(eta)
    x = np.diag(np.sqrt(np.arange((prms.maxPhotonNumber - 1)) + 1), -1) + np.diag(
        np.sqrt(np.arange((prms.maxPhotonNumber - 1)) + 1), +1)
    gk = - 2. * prms.t * np.sin(kVec) * eta
    eK = 2. * prms.t * np.cos(kVec)
    photonOne = np.diag(np.ones(prms.maxPhotonNumber))
    iHt = 1j * H[None, :, :] * tVec[:, None, None]
    iHtSinCos = - iHt \
                - 1j * gk[:, None, None, None] * x[None, None, :, :] * tVec[None, :, None, None] \
                - 1j * eK[:, None, None, None] * tVec[None, :, None, None] * photonOne[None, None, :, :]


    GF = np.zeros((len(kVec), len(tVec)), dtype='complex')
    for tInd in range(len(tVec)):
        prod1 = np.dot(phGS, sciLin.expm(iHt[tInd, :, :]))
        for kInd in range(len(kVec)):
            prod2 = np.dot(sciLin.expm(iHtSinCos[kInd, tInd, :, :]), phGS)
            GF[kInd, tInd] += np.dot(prod1, prod2)


    return - 1j * GF

def gfNumVecT(kVec, tVec, eta):
    initialState = np.zeros(prms.chainLength + 1, dtype='double')
    gs = anaGS.findGS1st(initialState, eta)
    _, occupations = np.meshgrid(np.ones(tVec.shape), gs[0: -1])
    GF = gfNumPointT(kVec, tVec, eta)
    GF = np.multiply(1 - occupations, GF)

    return GF


def getPhGSH1(eta):
    initialState = np.zeros(prms.chainLength + 1, dtype='double')
    gs = anaGS.findGS1st(initialState, eta)
    gsJ = eF.J(gs[0: -1])
    gsT = eF.T(gs[0: -1])
    return phState.findPhotonGS([gsT, gsJ], eta, 1)


def getH1(eta):
    initialState = np.zeros(prms.chainLength + 1, dtype='double')
    gs = anaGS.findGS1st(initialState, eta)
    gsJ = eF.J(gs[0: -1])
    gsT = eF.T(gs[0: -1])
    return numH.setupPhotonHamiltonian1st(gsT, gsJ, eta)


def calcSpectralPoint(kPoint, wPoint, eta):
    initialState = np.zeros(prms.chainLength + 1, dtype='double')
    gs = anaGS.findGS(initialState, eta)
    T = eF.T(gs[0: prms.chainLength])
    J = eF.J(gs[0: prms.chainLength])
    wDash = prms.w0 - eta ** 2 * T
    gK = - 2. * eta * prms.t * np.sin(kPoint)
    eK = 2. * prms.t * np.cos(kPoint)
    eKBar = eK - gK * gK / wDash

    deltaPlus = 1e-8

    ellCutoff = 10

    spectral = 0.
    for ell in range(ellCutoff):
        lorentz = deltaPlus / ((wPoint - (eKBar - 2. * (gK / wDash) * J + ell * wDash)) ** 2 + deltaPlus ** 2)
        ellFac = 1. / (np.math.factorial(ell)) * (gK / wDash) ** (2 * ell)
        expPrefac = np.exp(- gK ** 2 / (wDash ** 2))
        spectral += expPrefac * ellFac * lorentz * deltaPlus
        # lorentz = deltaPlus / ((wPoint - eK)**2 + deltaPlus**2)
        # spectral += lorentz

    return np.log10(spectral)


def calcSpectral(kVec, wVec, eta):
    kVec, wVec = np.meshgrid(kVec, wVec)

    spectralFunc = calcSpectralPoint(kVec, wVec, eta)

    return spectralFunc

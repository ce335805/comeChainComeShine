import numpy as np
import sec_order.analyticalEGS as anaGS
import globalSystemParams as prms
import energyFunctions as eF
from arb_order import photonState as phState
import scipy.linalg as sciLin


def g0T(kPoint, tPoint):
    return - 1j * np.exp(-1j * 2. * prms.t * np.cos(kPoint[:, None]) * tPoint[None, :])


def g0VecT(kVec, tVec):
    initialState = np.zeros(prms.chainLength + 1, dtype='double')
    gs = anaGS.findGS1st(initialState, 0.)
    _, occupations = np.meshgrid(np.ones(tVec.shape), gs[0: -1])
    GF = g0T(kVec, tVec)
    GF = np.multiply(1 - occupations, GF)
    return GF


def anaGreenPointT(kPoint, tPoint, gsJ, eta):
    epsK = 2. * prms.t * np.cos(kPoint[:, None])
    coupling = - eta ** 2 / prms.w0 * \
               (2. * (-2. * gsJ * prms.t * np.sin(kPoint[:, None])) + (-2. * prms.t * np.sin(kPoint[:, None]))**2)

    eTime = -1j * epsK * tPoint[None, :] - 1j * coupling * tPoint[None, :]
    ptTime = - (- 2. * eta * prms.t * np.sin(kPoint[:, None]))**2 / prms.w0**2 * (1. - np.exp(-1j * prms.w0 * tPoint[None, :]))
    return -1j * np.exp(eTime + ptTime)


def anaGreenVecT(kVec, tVec, eta):
    initialState = np.zeros(prms.chainLength + 1, dtype='double')
    gs = anaGS.findGS1st(initialState, eta)
    gsJ = eF.J(gs[0: -1])
    _, occupations = np.meshgrid(np.ones(tVec.shape), gs[0: -1])
    GF = anaGreenPointT(kVec, tVec, gsJ, eta)
    GF = np.multiply(1 - occupations, GF)
    print("GF.shape = {}".format(GF.shape))

    return GF


def gfNumPointT(kVec, tVec, eta):
    print("calculating GF numrically")

    phGS = getPhGS(eta)
    print(phGS)

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


    expiHt = np.zeros(iHt.shape, dtype='complex')
    for t in range(iHt.shape[0]):
        expiHt[t, :, :] = sciLin.expm(iHt[t, :, :])
    expiHtSinCos = np.zeros(iHtSinCos.shape, dtype='complex')
    for k in range(iHtSinCos.shape[0]):
        for t in range(iHtSinCos.shape[1]):
            expiHtSinCos[k, t] = sciLin.expm(iHtSinCos[k, t, :, :])

    print("expiHt.shape = {}".format(expiHt.shape))

    expHH = np.zeros((len(kVec), len(tVec), prms.maxPhotonNumber, prms.maxPhotonNumber), dtype='complex')
    for kInd in range(len(kVec)):
        for tInd in range(len(tVec)):
            expHH[kInd, tInd, :, :] = np.dot(expiHt[tInd, :, :], expiHtSinCos[kInd, tInd, :, :])

    GF = np.dot(np.dot(phGS, expHH), phGS)

    return - 1j * GF


def gfNumVecT(kVec, tVec, eta):
    initialState = np.zeros(prms.chainLength + 1, dtype='double')
    gs = anaGS.findGS1st(initialState, eta)
    _, occupations = np.meshgrid(np.ones(tVec.shape), gs[0: -1])
    GF = gfNumPointT(kVec, tVec, eta)
    GF = np.multiply(1 - occupations, GF)

    return GF


def getPhGS(eta):
    initialState = np.zeros(prms.chainLength + 1, dtype='double')
    gs = anaGS.findGS1st(initialState, eta)
    gsJ = eF.J(gs[0: -1])
    gsT = eF.T(gs[0: -1])
    return phState.findPhotonGS([gsT, gsJ], eta)


def getH1(eta):
    initialState = np.zeros(prms.chainLength + 1, dtype='double')
    gs = anaGS.findGS1st(initialState, eta)
    gsJ = eF.J(gs[0: -1])
    gsT = eF.T(gs[0: -1])
    return phState.setupPhotonHamiltonian1st(gsT, gsJ, eta)


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

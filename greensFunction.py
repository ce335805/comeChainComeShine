import numpy as np
import sec_order.analyticalEGS as anaGS
import globalSystemParams as prms
import energyFunctions as eF
from arb_order import photonState as phState
from arb_order import numHamiltonians as numH
import scipy.linalg as sciLin


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
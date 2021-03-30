import numpy as np
import globalSystemParams as prms
import energyFunctions as eF
import scipy.linalg as sciLin

from arb_order import arbOrder
from arb_order import photonState as phState

def currentGS(eta):
    initialState = np.zeros(prms.chainLength, dtype='double')
    initialState[ : prms.numberElectrons // 2 + 1] = 1.0
    initialState[ prms.chainLength - prms.numberElectrons//2 - 0: ] = 1.0
    print(initialState)
    gs = arbOrder.findGS(initialState, eta, 3)
    gsJ = eF.J(gs)
    gsT = eF.T(gs)

    gsE = phState.energyFromState(gs, eta, 3)
    print("gsE = {}".format(gsE - .5))

    ptGS = phState.findPhotonGS([gsT, gsJ], eta, 3)
    jOp = setupCurrentOperator(gsT, gsJ, eta)

    current = np.dot(np.conj(ptGS), np.dot(jOp, ptGS))
    return current

def expectA(eta):
    initialState = np.zeros(prms.chainLength, dtype='double')
    initialState[0: prms.numberElectrons] = 1.0
    gs = arbOrder.findGS(initialState, eta, 3)
    gsJ = eF.J(gs)
    gsT = eF.T(gs)

    ptGS = phState.findPhotonGS([gsT, gsJ], eta, 3)
    x = setUpA(eta)

    return np.dot(np.conj(ptGS), np.dot(x, ptGS))


def expectCosA(eta):
    initialState = np.zeros(prms.chainLength, dtype='double')
    initialState[0: prms.numberElectrons] = 1.0
    gs = arbOrder.findGS(initialState, eta, 3)
    gsJ = eF.J(gs)
    gsT = eF.T(gs)

    ptGS = phState.findPhotonGS([gsT, gsJ], eta, 3)
    x = setUpA(eta)
    cosX = sciLin.cosm(x)

    return np.dot(np.conj(ptGS), np.dot(cosX, ptGS))

def expectAnnihil(eta):
    initialState = np.zeros(prms.chainLength, dtype='double')
    initialState[0: prms.numberElectrons] = 1.0
    gs = arbOrder.findGS(initialState, eta, 3)
    gsJ = eF.J(gs)
    gsT = eF.T(gs)

    ptGS = phState.findPhotonGS([gsT, gsJ], eta, 3)
    a = setUpAnnihil(eta)

    return np.dot(np.conj(ptGS), np.dot(a, ptGS))

def expectSinA(eta):
    initialState = np.zeros(prms.chainLength, dtype='double')
    initialState[0: prms.numberElectrons] = 1.0
    gs = arbOrder.findGS(initialState, eta, 3)
    gsJ = eF.J(gs)
    gsT = eF.T(gs)

    ptGS = phState.findPhotonGS([gsT, gsJ], eta, 3)
    x = setUpA(eta)
    sinX = sciLin.sinm(x)

    return np.dot(np.conj(ptGS), np.dot(sinX, ptGS))

def setupCurrentOperator(T, J, eta):
    x = eta * (np.diag(np.sqrt(np.arange((prms.maxPhotonNumber - 1)) + 1), -1) + np.diag(
        np.sqrt(np.arange((prms.maxPhotonNumber - 1)) + 1), +1))
    sinX = sciLin.sinm(x)
    cosX = sciLin.cosm(x)
    return cosX * J - sinX * T

def setUpA(eta):
    x = eta * (np.diag(np.sqrt(np.arange((prms.maxPhotonNumber - 1)) + 1), -1) + np.diag(
        np.sqrt(np.arange((prms.maxPhotonNumber - 1)) + 1), +1))
    return x


def setUpAnnihil(eta):
    a = np.diag(np.sqrt(np.arange((prms.maxPhotonNumber - 1)) + 1), +1)
    return a
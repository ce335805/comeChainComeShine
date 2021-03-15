import numpy as np
import globalSystemParams as prms
from scipy.linalg import sinm
from scipy.linalg import cosm

def electronNumberZero(state):
    return np.sum(state[0: prms.chainLength]) - prms.numberElectrons


def calcAplusAdagger():
    offDiagonal = np.sqrt(np.arange((prms.maxPhotonNumber - 1)) + 1)
    return np.diag(offDiagonal, -1) + np.diag(offDiagonal, +1)

def calcSinAdaggerA(eta):
    aDaggerA = eta * calcAplusAdagger()
    return sinm(aDaggerA)[0 : prms.maxPhotonNumber, 0 : prms.maxPhotonNumber]

def calcCosAdaggerA(eta):
    aDaggerA = eta * calcAplusAdagger()
    return cosm(aDaggerA)[0 : prms.maxPhotonNumber, 0 : prms.maxPhotonNumber]

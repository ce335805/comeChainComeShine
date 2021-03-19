import numpy as np
import globalSystemParams as prms
import utils

def setupPhotonHamiltonian1st(T, J, eta):
    indices = np.arange(prms.maxPhotonNumber)
    diagonal = prms.w0 * (indices + .5) + T
    off1Diagonal = eta * J * np.sqrt(np.arange((prms.maxPhotonNumber - 1)) + 1)
    hamiltonian = np.diag(diagonal, 0) + np.diag(off1Diagonal, 1) + np.diag(off1Diagonal, -1)

    return hamiltonian


def setupPhotonHamiltonian2nd(T, J, eta):
    # return setupPhotonHamiltonian1st(T, J, eta)
    indices = np.arange(prms.maxPhotonNumber)
    diagonal = prms.w0 * (indices + .5) - eta * eta * (indices + .5) * T + T
    off1Diagonal = eta * J * np.sqrt(np.arange((prms.maxPhotonNumber - 1)) + 1)
    off2Diagonal = - .5 * eta * eta * T * np.sqrt(np.arange((prms.maxPhotonNumber - 2)) + 1) * np.sqrt(
        np.arange((prms.maxPhotonNumber - 2)) + 2)
    hamiltonian = np.diag(diagonal, 0) + np.diag(off1Diagonal, 1) + np.diag(off1Diagonal, -1) + \
                  np.diag(off2Diagonal, 2) + np.diag(off2Diagonal, -2)

    return hamiltonian


def setupPhotonHamiltonianInf(T, J, eta):
    #return setupPhotonHamiltonian2nd(T, J, eta)
    indices = np.arange(prms.maxPhotonNumber)
    photonEnergies = prms.w0 * np.diag((indices + .5), 0)
    jTerm = J * utils.calcSinAdaggerA(eta)
    tTerm =  T * utils.calcCosAdaggerA(eta)
    return photonEnergies + jTerm + tTerm
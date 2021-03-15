import numpy as np
import globalSystemParams as prms
from scipy.linalg import eigh
from scipy.linalg import sinm
from scipy.linalg import cosm
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


def setupPhotonHamiltonian4th(T, J, eta):
    indices = (1. + 0j) * np.arange(prms.maxPhotonNumber)
    indices1Off = (1. + 0j) * np.arange(prms.maxPhotonNumber - 1)
    indices2Off = (1. + 0j) * np.arange(prms.maxPhotonNumber - 2)
    diagonal = prms.w0 * (indices + .5) + (1. - .5 * eta * eta) * T - eta * eta * indices * T + \
               1. / (4.) * eta * eta * T * (indices * (indices + 1) + .5)
    off1Diagonal = eta * J * np.ones((prms.maxPhotonNumber - 1), dtype='complex') - \
                   1. / 2. * eta ** 3 * J * (indices1Off + 1.)
    off2Diagonal = - 0.5 * eta ** 2 * T * \
                   np.ones((prms.maxPhotonNumber - 2), dtype='complex') + \
                   1. / 6. * eta ** 4 * T * (indices2Off + 1.5)
    off3Diagonal = -1. / 6. * eta ** 3 * J * np.ones((prms.maxPhotonNumber - 3), dtype='complex')
    off4Diagonal = 1. / 24. * eta ** 4 * T * np.ones((prms.maxPhotonNumber - 4), dtype='complex')
    hamiltonian = np.diag(diagonal, 0) + \
                  np.diag(off1Diagonal, 1) + np.diag(off1Diagonal, -1) + \
                  np.diag(off2Diagonal, 2) + np.diag(off2Diagonal, -2) + \
                  np.diag(off3Diagonal, 3) + np.diag(off3Diagonal, -3) + \
                  np.diag(off4Diagonal, 4) + np.diag(off4Diagonal, -4)
    return hamiltonian


def findSmalestEigenvalue(TJ, eta):
    # hamiltonian = setupPhotonHamiltonian2nd(TJ[0], TJ[1], eta)
    hamiltonian = setupPhotonHamiltonianInf(TJ[0], TJ[1], eta)
    # hamiltonian = setupPhotonHamiltonian4th(TJ[0], TJ[1], eta)
    v = eigh(hamiltonian, eigvals_only=True, eigvals=[0, 0])
    return v[0]


def findPhotonGS(TJ, eta):
    # hamiltonian = setupPhotonHamiltonian2nd(TJ[0], TJ[1], eta)
    hamiltonian = setupPhotonHamiltonianInf(TJ[0], TJ[1], eta)
    # hamiltonian = setupPhotonHamiltonian4th(TJ[0], TJ[1], eta)
    # v, eVec = eigh(hamiltonian, eigvals_only=False, eigvals=[0, 0])
    v, eVec = np.linalg.eigh(hamiltonian)
    return eVec[0, :]


def averagePhotonNumber(TJ, eta):
    # hamiltonian = setupPhotonHamiltonian2nd(TJ[0], TJ[1], eta)
    hamiltonian = setupPhotonHamiltonianInf(TJ[0], TJ[1], eta)
    # hamiltonian = setupPhotonHamiltonian4th(TJ[0], TJ[1], eta)
    # v, eVec = eigh(hamiltonian, eigvals_only=False, eigvals=[0, 0])
    # eVec = eVec[:, 0]
    v, eVec = np.linalg.eigh(hamiltonian)
    eVec = eVec[0, :]
    occ = np.multiply(eVec, np.conj(eVec))
    avPhNum = np.sum(np.multiply(np.arange(len(eVec)), occ))
    return avPhNum

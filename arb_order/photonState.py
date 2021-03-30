import numpy as np
from scipy.linalg import eigh
from arb_order import numHamiltonians as numH
import energyFunctions as eF

def findSmalestEigenvalue(TJ, eta, orderH):
    hamiltonian = setupHOrder(TJ, eta, orderH)
    v = eigh(hamiltonian, eigvals_only=True, eigvals=[0, 0])
    return v[0]

def findPhotonGS(TJ, eta, orderH):
    hamiltonian = setupHOrder(TJ, eta, orderH)
    v, eVec = np.linalg.eigh(hamiltonian)
    return eVec[:, 0]

def averagePhotonNumber(TJ, eta, orderH):
    hamiltonian = setupHOrder(TJ, eta, orderH)

    v, eVec = np.linalg.eigh(hamiltonian)
    eVec = eVec[:, 0]
    occ = np.multiply(eVec, np.conj(eVec))
    avPhNum = np.sum(np.multiply(np.arange(len(eVec)), occ))
    return avPhNum

def setupHOrder(TJ, eta, orderH):
    if(orderH == 1):
        return numH.setupPhotonHamiltonian1st(TJ[0], TJ[1], eta)
    elif(orderH == 2):
        return numH.setupPhotonHamiltonian2nd(TJ[0], TJ[1], eta)
    else:
        return numH.setupPhotonHamiltonianInf(TJ[0], TJ[1], eta)

def energyFromState(electronicState, eta, orderH):
    T = eF.T(electronicState)
    J = eF.J(electronicState)
    E = findSmalestEigenvalue([T, J], eta, orderH)
    return E

def photonGS(electronicState, eta, orderH):
    T = eF.T(electronicState)
    J = eF.J(electronicState)
    gs = findPhotonGS([T, J], eta, orderH)
    return gs

def avPhotonNum(electronicState, eta, orderH):
    T = eF.T(electronicState)
    J = eF.J(electronicState)
    av = averagePhotonNumber([T, J], eta, orderH)
    return av
import numpy as np
from arb_order import numHamiltonians
import matplotlib.pyplot as plt
import globalSystemParams as prms


def plotPtOcc(tArr, eta):
    nArrInf = photonOccInf(tArr, eta)
    nArrSec = photonOccSec(tArr, eta)

    fig, ax = plt.subplots(nrows=1, ncols=1)
    ax.plot(np.abs(tArr), nArrInf, color = 'skyblue', label = '$\infty$-order')
    ax.plot(np.abs(tArr), nArrSec, color = 'gray', linestyle = '--', label = '$2$-order')
    ax.set_xlabel('T')
    ax.set_ylabel('$N_{pt}$')
    plt.legend()
    plt.show()

def photonOccInf(tArr, eta):

    nArr = np.zeros((len(tArr)))
    for indT in range(len(tArr)):
        T = tArr[indT]
        etaEff = eta / np.sqrt(np.abs(T))
        H = numHamiltonians.setupPhotonHamiltonianInf(T, 0., etaEff)
        eVal, eVecs = np.linalg.eigh(H)

        indices = np.arange(prms.maxPhotonNumber)
        eVec0 = eVecs[:, 0]
        temp = np.multiply(indices, eVec0)
        nPt = np.dot(np.conj(eVec0), temp)

        nArr[indT] = nPt
    return nArr

def photonOccSec(tArr, eta):

    nArr = np.zeros((len(tArr)))
    for indT in range(len(tArr)):
        T = tArr[indT]
        etaEff = eta / np.sqrt(np.abs(T))
        H = numHamiltonians.setupPhotonHamiltonian2nd(T, 0., etaEff)
        eVal, eVecs = np.linalg.eigh(H)

        indices = np.arange(prms.maxPhotonNumber)
        eVec0 = eVecs[:, 0]
        temp = np.multiply(indices, eVec0)
        nPt = np.dot(np.conj(eVec0), temp)

        nArr[indT] = nPt
    return nArr
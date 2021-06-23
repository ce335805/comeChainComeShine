import numpy as np
from arb_order import arbOrder
import globalSystemParams as prms
from arb_order import numHamiltonians
import energyFunctions as eF
from arb_order import photonState as phState
from automatedTests import testUtils as util
import scipy.linalg as sciLin

def matrixDiagGEMM():

    eta = .2
    gsT = 10.
    gsJ = 1.

    H = numHamiltonians.setupPhotonHamiltonianInf(gsT, gsJ, eta)

    eVals, eVecs = np.linalg.eigh(H)
    HDiag1 = np.diag(eVals)
    HDiag2 = np.dot(np.transpose(np.conj(eVecs)), np.dot(H, eVecs))

    diff = np.sum(np.abs(HDiag1 - HDiag2))
    fail = diff > 1e-10
    if (fail):
        print(" Diagonalization with matrix multiplication failed!!! ------ CHECK FAILED!!!")
        return False
    else:
        print(" Diagonalization via matrix multiplication worked! ------ CHECK PASSED :)")
        return True

def matrixExpGEMM():

    eta = .2
    gsT = 10.
    gsJ = 1.

    H = numHamiltonians.setupPhotonHamiltonianInf(gsT, gsJ, eta)

    expH1 = sciLin.expm(H)

    eVals, eVecs = np.linalg.eigh(H)
    expH2 = np.dot(eVecs, np.dot(np.diag(np.exp(eVals)), np.transpose(np.conj(eVecs))))

    #print(np.abs(expH1 - expH2))

    diff = np.sum(np.abs(expH1 - expH2)) / (np.sum(np.abs(expH1)))
    print("rel diff exp summed = {}".format(diff))
    fail = diff > 1e-5
    if (fail):
        print(" Diagonalization with matrix multiplication failed!!! ------ CHECK FAILED!!!")
        return False
    else:
        print(" Diagonalization via matrix multiplication worked! ------ CHECK PASSED :)")
        return True


def runAllTests():
    check1 = matrixDiagGEMM()
    check2 = matrixExpGEMM()

    print("---------------------------")
    print("--- GS tests finished! ---")
    print("---------------------------")
    success = check1 and check2
    util.printSuccessMessage(success)
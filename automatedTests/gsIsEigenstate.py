import numpy as np
from arb_order import arbOrder
import globalSystemParams as prms
from arb_order import numHamiltonians
import energyFunctions as eF
from arb_order import photonState as phState
from automatedTests import testUtils as util
import initialState as ini

def eighWorksAsExpected():

    eta = .5

    eGS = arbOrder.findGS(eta, 1)
    gsJ = eF.J(eGS)
    gsT = eF.T(eGS)

    H1 = numHamiltonians.setupPhotonHamiltonian1st(gsT, gsJ, eta)
    v, eVec = np.linalg.eigh(H1)

    E0_1 = v[0]
    E0_2 = np.dot(np.conj(eVec[:, 0]), np.dot(H1, eVec[:, 0]))

    diff = np.abs(E0_1 - E0_2)
    fail = diff > 1e-10
    if (fail):
        print(" Smallest e-val NOT obtained through dot!!! ------ CHECK FAILED!!!")
        return False
    else:
        print(" Smallest e-val also obtained through dot! ------ CHECK PASSED :)")
        return True


def photonGSIsEigenstate():

    eta = .5

    eGS = arbOrder.findGS(eta, 1)
    gsJ = eF.J(eGS)
    gsT = eF.T(eGS)

    ptGS = phState.findPhotonGS([gsT, gsJ], eta, 1)
    gsEnergy = phState.energyFromState(eGS, eta, 1)
    H1 = numHamiltonians.setupPhotonHamiltonian1st(gsT, gsJ, eta)


    diff = np.abs(gsEnergy - np.dot(np.conj(ptGS), np.dot(H1, ptGS)))
    fail = diff > 1e-10
    if (fail):
        print(" Smallest eval NOT consistent with dot evaluation!!! ------ CHECK FAILED!!!")
        return False
    else:
        print(" Smallest eval consistent with dot evaluation! ------ CHECK PASSED :)")
        return True

def runAllTests():
    check1 = eighWorksAsExpected()
    check2 = photonGSIsEigenstate()


    print("---------------------------")
    print("--- GS tests finished! ---")
    print("---------------------------")
    success = check1 and check2
    util.printSuccessMessage(success)
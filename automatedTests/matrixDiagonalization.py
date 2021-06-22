import numpy as np
from arb_order import arbOrder
import globalSystemParams as prms
from arb_order import numHamiltonians
import energyFunctions as eF
from arb_order import photonState as phState
from automatedTests import testUtils as util
import scipy.linalg as sciLin


def eighWorksAsExpected():

    eta = .2

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
        print(" Smallest e-val NOT obtained through dot 1st!!! ------ CHECK FAILED!!!")
        return False
    else:
        print(" Smallest e-val also obtained through dot 1st! ------ CHECK PASSED :)")
        return True


def photonGSIsEigenstate1st():

    eta = .2

    eGS = arbOrder.findGS(eta, 1)
    gsJ = eF.J(eGS)
    gsT = eF.T(eGS)

    ptGS = phState.findPhotonGS([gsT, gsJ], eta, 1)
    gsEnergy = phState.energyFromState(eGS, eta, 1)
    H1 = numHamiltonians.setupPhotonHamiltonian1st(gsT, gsJ, eta)


    diff = np.abs(gsEnergy - np.dot(np.conj(ptGS), np.dot(H1, ptGS)))
    fail = diff > 1e-10
    if (fail):
        print(" Smallest eval NOT consistent with dot evaluation 2nd!!! ------ CHECK FAILED!!!")
        return False
    else:
        print(" Smallest eval consistent with dot evaluation 2nd! ------ CHECK PASSED :)")
        return True

def photonGSIsEigenstate2nd():

    eta = .2

    eGS = arbOrder.findGS(eta, 2)
    gsJ = eF.J(eGS)
    gsT = eF.T(eGS)

    ptGS = phState.findPhotonGS([gsT, gsJ], eta, 2)
    gsEnergy = phState.energyFromState(eGS, eta, 2)
    H2 = numHamiltonians.setupPhotonHamiltonian2nd(gsT, gsJ, eta)


    diff = np.abs(gsEnergy - np.dot(np.conj(ptGS), np.dot(H2, ptGS)))
    fail = diff > 1e-10
    if (fail):
        print(" Smallest eval NOT consistent with dot evaluation Inf!!! ------ CHECK FAILED!!!")
        return False
    else:
        print(" Smallest eval consistent with dot evaluation Inf! ------ CHECK PASSED :)")
        return True

def photonGSIsEigenstateInf():

    eta = .2

    eGS = arbOrder.findGS(eta, 3)
    gsJ = eF.J(eGS)
    gsT = eF.T(eGS)

    ptGS = phState.findPhotonGS([gsT, gsJ], eta, 3)
    gsEnergy = phState.energyFromState(eGS, eta, 3)
    H = numHamiltonians.setupPhotonHamiltonianInf(gsT, gsJ, eta)


    diff = np.abs(gsEnergy - np.dot(np.conj(ptGS), np.dot(H, ptGS)))
    fail = diff > 1e-10
    if (fail):
        print(" Smallest eval NOT consistent with dot evaluation!!! ------ CHECK FAILED!!!")
        return False
    else:
        print(" Smallest eval consistent with dot evaluation! ------ CHECK PASSED :)")
        return True




def takeingE0inExpWorks():

    eta = .3

    eGS = arbOrder.findGS(eta, 3)
    gsJ = eF.J(eGS)
    gsT = eF.T(eGS)

    ptGS = phState.findPhotonGS([gsT, gsJ], eta, 3)
    gsEnergy = phState.energyFromState(eGS, eta, 3)
    H = numHamiltonians.setupPhotonHamiltonianInf(gsT, gsJ, eta)

    expiH = np.dot(np.conj(ptGS), np.dot(sciLin.expm(1j * H), ptGS))
    expiE = np.exp(1j * gsEnergy)

    print("<gs| exp(iH) |gs> = {}".format(expiH))
    print("exp(iE0) = {}".format(expiE))
    diff = np.abs(expiH - expiE)
    fail = diff > 1e-10
    if (fail):
        print(" expH not consisten with expE!!! ------ CHECK FAILED!!!")
        return False
    else:
        print(" expH is consisten with expE! ------ CHECK PASSED :)")
        return True

def runAllTests():
    check1 = eighWorksAsExpected()
    check2 = photonGSIsEigenstate1st()
    check3 = photonGSIsEigenstate2nd()
    check4 = photonGSIsEigenstateInf()
    check5 = takeingE0inExpWorks()

    print("---------------------------")
    print("--- GS tests finished! ---")
    print("---------------------------")
    success = check1 and check2 and check3 and check4 and check5
    util.printSuccessMessage(success)
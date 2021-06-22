import numpy as np
from arb_order import arbOrder as numerical
from sec_order import analyticalEGS as analytical
import globalSystemParams as prms
from automatedTests import testUtils as util


def electronicGSMatches1st():
    eta = 1. / np.sqrt(prms.chainLength)

    gsAna = analytical.findGS1st(eta)
    gsAna = gsAna[:-1]

    gsNum = numerical.findGS(eta, 1)

    failArr = (np.abs(gsAna - gsNum) > 1e-8)

    #print("gsAna = {}".format(gsAna))
    #print("gsNum = {}".format(gsNum))

    if (np.any(failArr)):
        print(" Numerical and Analytical result differ for GS 1st!!! ------ CHECK FAILED!!!")
        return False
    else:
        print(" Numerical and Analytical result consistent for GS 1st! ------ CHECK PASSED :)")
        return True

def electronicGSMatches2nd():
    eta = .5 / np.sqrt(prms.chainLength)

    gsAna = analytical.findGSExactSec(eta)
    gsAna = gsAna[:-1]

    gsNum = numerical.findGS(eta, 2)

    print("gsAna = {}".format(gsAna))
    print("gsNum = {}".format(gsNum))

    failArr = (np.abs(gsAna - gsNum) > 1e-8)

    if (np.any(failArr)):
        print(" Numerical and Analytical result differ for GS 2nd!!! ------ CHECK FAILED!!!")
        return False
    else:
        print(" Numerical and Analytical result consistent for GS 2nd! ------ CHECK PASSED :)")
        return True

def gsEnergyMatches1st():
    etas = [.0, .1, .2]

    eAna = analytical.findGSEnergy1st(etas)

    eNum = numerical.findGSEnergies(etas, 1)

    diffArr = eAna - eNum
    print(diffArr)
    failArr = (np.abs(eAna - eNum) > 1e-8)

    if (np.any(failArr)):
        print(" Numerical and Analytical result differ for GS Energy 1st!!! ------ CHECK FAILED!!!")
        return False
    else:
        print(" Numerical and Analytical result consistent for GS Energy 1st! ------ CHECK PASSED :)")
        return True

def gsEnergyMatches2nd():
    etas = [.0, .05, .1, .2]

    eAna = analytical.findGSEnergyExactSec(etas)

    eNum = numerical.findGSEnergies(etas, 2)

    diffArr = eAna - eNum
    print(diffArr)
    failArr = (np.abs(eAna - eNum) > 1e-8)

    if (np.any(failArr)):
        print(" Numerical and Analytical result differ for GS Energy 2nd!!! ------ CHECK FAILED!!!")
        return False
    else:
        print(" Numerical and Analytical result consistent for GS Energy 2nd! ------ CHECK PASSED :)")
        return True

def gsPhotonNumberMatches1st():
    etas = [.0, .05, .1, .2]

    phAna = analytical.findPhotonNumber1st(etas)

    phNum = numerical.findPhotonNumbers(etas, 1)

    failArr = (np.abs(phAna - phNum) > 1e-8)

    diffArr = np.abs(phAna - phNum)
    print(diffArr)
    if (np.any(failArr)):
        print(" Numerical and Analytical photon number differ in 1st!!! ------ CHECK FAILED!!!")
        return False
    else:
        print(" Numerical and Analytical photon number consistent in 1st! ------ CHECK PASSED :)")
        return True


def gsPhotonNumberMatches2nd():
    etas = np.array([.0, .1, .2, .5]) / np.sqrt(prms.chainLength)

    phAna = analytical.findPhotonNumberExactSec(etas)

    phNum = numerical.findPhotonNumbers(etas, 2)

    failArr = (np.abs(phAna - phNum) > 1e-8)

    diffArr = np.abs(phAna - phNum)
    print(diffArr)
    if (np.any(failArr)):
        print(" Numerical and Analytical photon number differ in 2nd!!! ------ CHECK FAILED!!!")
        return False
    else:
        print(" Numerical and Analytical photon number consistent in 2nd! ------ CHECK PASSED :)")
        return True

def runAllTests():
    check1 = electronicGSMatches1st()
    check2 = electronicGSMatches2nd()
    check3 = gsEnergyMatches1st()
    check4 = gsEnergyMatches2nd()
    check5 = gsPhotonNumberMatches1st()
    check6 = gsPhotonNumberMatches2nd()

    print("---------------------------")
    print("--- GS tests finished! ---")
    print("---------------------------")
    success = check1 and check2 and check3 and check4 and check5 and check6
    util.printSuccessMessage(success)

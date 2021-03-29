import numpy as np
from arb_order import arbOrder as numerical
from sec_order import analyticalEGS as analytical
import globalSystemParams as prms
from automatedTests import testUtils as util


def electronicGSMatches1st():
    eta = .3

    iniAna = np.zeros(prms.chainLength + 1, dtype='double')
    iniAna[0: prms.numberElectrons] = 1.0
    gsAna = analytical.findGS1st(iniAna, eta)
    gsAna = gsAna[:-1]

    iniNum = np.zeros(prms.chainLength, dtype='double')
    iniNum[0: prms.numberElectrons] = 1.0
    gsNum = numerical.findGS(iniNum, eta, 1)

    failArr = (np.abs(gsAna - gsNum) > 1e-8)

    if (np.any(failArr)):
        print(" Numerical and Analytical result differ for GS 1st!!! ------ CHECK FAILED!!!")
        return False
    else:
        print(" Numerical and Analytical result consistent for GS 1st! ------ CHECK PASSED :)")
        return True

def electronicGSMatches2nd():
    eta = .2

    iniAna = np.zeros(prms.chainLength + 1, dtype='double')
    iniAna[0: prms.numberElectrons] = 1.0
    gsAna = analytical.findGSExactSec(iniAna, eta)
    gsAna = gsAna[:-1]

    iniNum = np.zeros(prms.chainLength, dtype='double')
    iniNum[0: prms.numberElectrons] = 1.0
    gsNum = numerical.findGS(iniNum, eta, 2)

    failArr = (np.abs(gsAna - gsNum) > 1e-8)

    if (np.any(failArr)):
        print(" Numerical and Analytical result differ for GS 2nd!!! ------ CHECK FAILED!!!")
        return False
    else:
        print(" Numerical and Analytical result consistent for GS 2nd! ------ CHECK PASSED :)")
        return True

def gsEnergyMatches1st():
    etas = [.0, .1, .2, .3]

    iniAna = np.zeros(prms.chainLength + 1, dtype='double')
    iniAna[0: prms.numberElectrons] = 1.0
    eAna = analytical.findGSEnergy1st(etas)

    iniNum = np.zeros(prms.chainLength, dtype='double')
    iniNum[0: prms.numberElectrons] = 1.0
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

    iniAna = np.zeros(prms.chainLength + 1, dtype='double')
    iniAna[0: prms.numberElectrons] = 1.0
    eAna = analytical.findGSEnergyExactSec(etas)

    iniNum = np.zeros(prms.chainLength, dtype='double')
    iniNum[0: prms.numberElectrons] = 1.0
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
    etas = [.0, .05, .1, .2, .3]


    iniAna = np.zeros(prms.chainLength + 1, dtype='double')
    iniAna[0: prms.numberElectrons] = 1.0
    phAna = analytical.findPhotonNumber1st(etas)

    iniNum = np.zeros(prms.chainLength, dtype='double')
    iniNum[0: prms.numberElectrons] = 1.0
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
    etas = [.0, .05, .1, .2]


    iniAna = np.zeros(prms.chainLength + 1, dtype='double')
    iniAna[0: prms.numberElectrons] = 1.0
    phAna = analytical.findPhotonNumberExactSec(etas)

    iniNum = np.zeros(prms.chainLength, dtype='double')
    iniNum[0: prms.numberElectrons] = 1.0
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

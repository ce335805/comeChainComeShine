import numpy as np
import greensFunction as green
import greenAna1st as greenAna
import green0
import globalSystemParams as prms
from automatedTests import testUtils as util

def g1stEQg0():

    eta = 0.0
    kVec = np.linspace(0, 2. * np.pi, prms.chainLength)
    tVec = np.linspace(0., 10. , 10)
    gfT = greenAna.anaGreenVecTGreater(kVec, tVec, eta, 0.)
    g0T = green0.g0VecT(kVec, tVec)

    failArr = (np.abs(gfT - g0T) > prms.accuracy)

    if(np.any(failArr)):
        print("G 1st not consistent with G0!!! ------ CHECK FAILED!!!")
        return False
    else:
        print("G 1st consistent with G0! ------ CHECK PASSED :)")
        return True

def g1NumEQg1Ana():
    eta = .1
    kVec = np.linspace(0, 2. * np.pi, prms.chainLength)
    tVec = np.linspace(0., 10. , 10)

    gfAna = greenAna.anaGreenVecTGreater(kVec, tVec, eta, 0.)
    gfNum = green.gfNumVecT(kVec, tVec, eta)

    failArr = (np.abs(gfAna - gfNum) > prms.accuracy)

    if(np.any(failArr)):
        print("Numerical GF1 not consistent with analytical GF1!!! ------ CHECK FAILED!!!")
        return False
    else:
        print("Numerical GF1 consistent with analytical GF1! ------ CHECK PASSED :)")
        return True

def runAllTests():
    check1 = g1stEQg0()
    check2 = g1NumEQg1Ana()

    print("---------------------------")
    print("--- Green's function tests finished! ---")
    print("---------------------------")
    success = check1 and check2
    util.printSuccessMessage(success)
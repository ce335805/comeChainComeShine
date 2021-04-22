import numpy as np
from floquet import floquetSpecPoint
from floquet import spectralFunction

import globalSystemParams as prms
from automatedTests import testUtils as util

def vecAndPointVersionsMatch():
    eta = .1
    kVec = np.array([1.1234])
    wVec = np.linspace(-10, 10, 10, endpoint=False)
    tAv  = np.linspace(0., 10., 10, endpoint=False)

    gfNonEq = spectralFunction.gLesserW(kVec, wVec, tAv, eta, 0.1, 1.)
    gfNonEqPoint = floquetSpecPoint.gLesserW(kVec[0], wVec, tAv, eta, 1., 0.1)

    failArr = (np.abs(gfNonEq - gfNonEqPoint) > prms.accuracy)

    if(np.any(failArr)):
        print("Floquet Vec is not consistent with Floquet Point!!! ------ CHECK FAILED!!!")
        return False
    else:
        print("Floquet Vec is consistent with Floquet Point! ------ CHECK Passed!!!")
        return True

def floquetRightIntegral():
    eta = .5
    kPoint = 7. * np.pi / 13.
    wVec = np.linspace(-10, 10, 100, endpoint=False)
    tAv = np.linspace(0., 10., 10, endpoint=False)

    gfNonEqPoint = floquetSpecPoint.gLesserW(kPoint, wVec, tAv, eta, 1., 0.1)

    dw = wVec[1] - wVec[0]
    intGS = np.sum(gfNonEqPoint[:, 0]) * dw

    print("int GF(w) = {}".format(-1j * intGS/np.sqrt(2. * np.pi)))

    return True

def runAllTests():
    check1 = vecAndPointVersionsMatch()
    check2 = floquetRightIntegral()

    print("---------------------------")
    print("--- Floquet tests finished! ---")
    print("---------------------------")
    success = check1 and check2
    util.printSuccessMessage(success)
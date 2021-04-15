import numpy as np
import globalSystemParams as prms
from greensFunction import greenNumArb
from nonEqGreen import nonEqGreen
from automatedTests import testUtils as util

def GreensEqual():
    eta = .4
    kVec = np.linspace(0, 2. * np.pi, prms.chainLength, endpoint=False)
    t = np.linspace(-50., 50., 20, endpoint = False)
    tRel = t
    tAv = t

    gfEq = greenNumArb.gfNumPointTGreater(kVec, t, eta)
    gfNonEq = nonEqGreen.gfPointTGS(kVec, tAv, tRel, eta)
    compGfNonEq = np.zeros((len(kVec), len(t)), dtype='complex')
    for indK in range(len(kVec)):
        compGfNonEq[indK, :] = np.diag(gfNonEq[indK, :, :])

    failArr = (np.abs(gfEq - compGfNonEq) > prms.accuracy)

    if(np.any(failArr)):
        print("G-Eq not consistent with G-Non-Eq!!! ------ CHECK FAILED!!!")
        return False
    else:
        print("G-Eq is consistent with G-Non-Eq! ------ CHECK PASSED :)")
        return True

def runAllTests():
    check1 = GreensEqual()

    print("---------------------------")
    print("--- Equilibrium vs non-Equilibrium tests finished! ---")
    print("---------------------------")
    success = check1
    util.printSuccessMessage(success)
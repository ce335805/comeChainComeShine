import numpy as np
import globalSystemParams as prms
from greensFunction import greenNumArb
from nonEqGreen import nonEqGreen
from automatedTests import testUtils as util

def GreensEqual():
    eta = .1
    kVec = np.linspace(0, 2. * np.pi, prms.chainLength, endpoint=False)
    t = np.linspace(-50., 50., 20, endpoint = False)
    tRel = t
    tAv = .5 * t

    gfEq = greenNumArb.gfNumPointTGreater(kVec, t, eta)
    gfNonEq = nonEqGreen.gfPointTGS(kVec, tRel, tAv, eta)
    compGfNonEq = np.zeros((len(kVec), len(t)), dtype='complex')
    for indK in range(len(kVec)):
        compGfNonEq[indK, :] = np.diag(gfNonEq[indK, :, :])

    print(compGfNonEq)
    print()
    print(gfEq)

    failArr = (np.abs(gfEq - compGfNonEq) > prms.accuracy)

    if(np.any(failArr)):
        print("G-Eq not consistent with G-Non-Eq!!! ------ CHECK FAILED!!!")
        return False
    else:
        print("G-Eq is consistent with G-Non-Eq! ------ CHECK PASSED :)")
        return True

def gsWNonEqRightIntegral():
    eta = .5 / np.sqrt(prms.chainLength)

    damping = .1
    kVec = np.linspace(0, 2. * np.pi, prms.chainLength, endpoint=False)
    wVec = np.linspace(-50., 50., 10000, endpoint=False)
    tAv = np.array([0.])
    # tAv = np.linspace(0., 1000., 3)

    gfNonEq = nonEqGreen.gfGSWLesser(kVec, wVec, tAv, eta, damping)
    gfNonEqCoh = nonEqGreen.gfCohWLesser(kVec, wVec, tAv, eta, damping, 3.)

    dw = wVec[1] - wVec[0]
    intGS = np.sum(gfNonEq[0, :, 0]) * dw
    intCoh = np.sum(gfNonEqCoh[0, :, 0]) * dw

    print("int GF(w) = {}".format(1j * intGS/np.sqrt(2. * np.pi)))
    print("int GF-Coh(w) = {}".format(1j* intCoh/np.sqrt(2. * np.pi)))

    return True

def runAllTests():
    check1 = GreensEqual()
    check2 = gsWNonEqRightIntegral()

    print("---------------------------")
    print("--- Equilibrium vs non-Equilibrium tests finished! ---")
    print("---------------------------")
    success = check1 and check2
    util.printSuccessMessage(success)
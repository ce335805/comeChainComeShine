import numpy as np
import globalSystemParams as prms
from greensFunction import greenNumArb
from nonEqGreen import nonEqGreen
from automatedTests import testUtils as util
from nonEqGreen import nonEqGreenPoint
import matplotlib.pyplot as plt


def GreensEqual():
    eta = 0.3 / np.sqrt(prms.chainLength)
    kVec = np.linspace(0, 2. * np.pi, prms.chainLength, endpoint=False)
    t = np.linspace(-50., 50., 10, endpoint = False)
    tRel = t
    tAv = .5 * t

    gfEq = greenNumArb.gfNumPointTGreater(kVec, t, eta)
    gfNonEq = nonEqGreen.gfPointTGS(kVec, tRel, tAv, eta)
    compGfNonEq = np.zeros((len(kVec), len(t)), dtype='complex')
    for indK in range(len(kVec)):
        compGfNonEq[indK, :] = np.diag(gfNonEq[indK, :, :])

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(t, np.imag(gfEq[8, :]), color = 'red')
    ax.plot(t, np.imag(compGfNonEq[8, :]), color = 'blue', linestyle = '--')
    plt.show()

    #print(compGfNonEq)
    #print()
    #print(gfEq)

    failArr = (np.abs(gfEq - compGfNonEq) > 1e-5)
    print("maxErr NonEq GF vs Eq GF = {}".format(np.amax(np.abs(gfEq - compGfNonEq))))

    if(np.any(failArr)):
        print("G-Eq not consistent with G-Non-Eq!!! ------ CHECK FAILED!!!")
        return False
    else:
        print("G-Eq is consistent with G-Non-Eq! ------ CHECK PASSED :)")
        return True

def compareEqAndNonEq():
    eta = .3 / np.sqrt(prms.chainLength)
    kVec = np.array([1.1234])
    kPoint = 1.1234
    wVec = np.linspace(-4, 4, 2000, endpoint=False)
    tAv = np.array([10.])
    damping = 0.1

    gfEq = greenNumArb.numGreenVecWGreater(kVec, wVec, eta, damping)
    gfNonEq = nonEqGreenPoint.gfGSWGreater(kPoint, wVec, tAv, eta, damping)

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(wVec, np.imag(gfEq[0, :]), color = 'red')
    ax.plot(wVec, np.imag(gfNonEq[:, 0]), color = 'blue', linestyle = '--')
    #ax.plot(wVec, np.imag(gfNonEq[:, 0] - gfEq[0, :]), color = 'blue', linestyle = '--')
    plt.show()

    #print(compGfNonEq)
    #print()
    #print(gfEq)

    failArr = (np.abs(gfEq[0, :] - gfNonEq[:, 0]) > 1e-5)
    print("maxErr NonEq GF vs Eq GF = {}".format(np.amax(np.abs(gfEq[0, :] - gfNonEq[:, 0]))))

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

def vecAndPointVersionsMatchCoh():
    eta = 1. / np.sqrt(prms.chainLength)
    kVec = np.array([1.1234])
    wVec = np.linspace(-10, 10, 10, endpoint=False)
    tAv  = np.linspace(0., 10., 10, endpoint=False)

    gfNonEq = nonEqGreen.gfCohWLesser(kVec, wVec, tAv, eta, 0.1, 1.)
    gfNonEqPoint = nonEqGreenPoint.gfCohWLesser(kVec[0], wVec, tAv, eta, 0.1, 1.)

    failArr = (np.abs(gfNonEq - gfNonEqPoint) > prms.accuracy)

    if(np.any(failArr)):
        print("G-Non-Eq-Vec non consistent with G-Non-Eq-Point - Coh!!! ------ CHECK FAILED!!!")
        return False
    else:
        print("G-Non-Eq-Vec is consistent with G-Non-Eq-Point - Coh! ------ CHECK Passed!!!")
        return True

def vecAndPointVersionsMatchGS():
    eta = .1
    kVec = np.array([1.1234])
    wVec = np.linspace(-10, 10, 10, endpoint=False)
    tAv  = np.linspace(0., 10., 10, endpoint=False)

    gfNonEq = nonEqGreen.gfGSWLesser(kVec, wVec, tAv, eta, .1)
    gfNonEqPoint = nonEqGreenPoint.gfGSWLesser(kVec[0], wVec, tAv, eta, 0.1)

    failArr = (np.abs(gfNonEq - gfNonEqPoint) > prms.accuracy)

    if(np.any(failArr)):
        print("G-Non-Eq-Vec non consistent with G-Non-Eq-Point - GS!!! ------ CHECK FAILED!!!")
        return False
    else:
        print("G-Non-Eq-Vec is consistent with G-Non-Eq-Point - GS! ------ CHECK Passed!!!")
        return True

def gfCohPHSymm():

    eta = 1. / np.sqrt(prms.chainLength)
    damping = 0.01
    N = 0.1
    kPoint1 = np.pi / 2. + 0.5
    kPoint2 = np.pi / 2. - 0.5
    wVec = np.linspace(-6, 6, 2000, endpoint=False)
    tAv  = np.linspace(100., 200., 200, endpoint=False)

    gf = nonEqGreenPoint.gfCohW(kPoint1, wVec, tAv, eta, damping, N)
    gfTurned = nonEqGreenPoint.gfCohWTurned(kPoint2, wVec, tAv, eta, damping, N)

    print("gf.shape = {}".format(gf.shape))

    gf = np.sum(gf, axis = 1) * (tAv[1] - tAv[0])
    gfTurned = np.sum(gfTurned, axis = 1) * (tAv[1] - tAv[0])

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(wVec, - np.imag(gf), color = 'red')
    ax.plot(-wVec, np.imag(gfTurned), color = 'blue', linestyle = '--')
    plt.show()

    failArr = (np.abs(gf - gfTurned) > 1.)

    if(np.any(failArr)):
        print("PH symmetry broken for Non-Eq GF!!! ------ CHECK FAILED!!!")
        return False
    else:
        print("PH symmetry works for Non-Eq GF! ------ CHECK Passed!!!")
        return True

def runAllTests():
    check1 = GreensEqual()
    check2 = gsWNonEqRightIntegral()
    check3 = vecAndPointVersionsMatchCoh()
    check4 = vecAndPointVersionsMatchGS()
    check5 = gfCohPHSymm()
    check6 = compareEqAndNonEq()

    print("---------------------------")
    print("--- Equilibrium vs non-Equilibrium tests finished! ---")
    print("---------------------------")
    #success = check1 and check3 and check4
    success = check1 and check3 and check4 and check2 and check5 and check6
    #success = check5
    util.printSuccessMessage(success)
import numpy as np
from greensFunction import green0, greenAna1st as greenAna, greenNum1st as green
import globalSystemParams as prms
from automatedTests import testUtils as util

import matplotlib.pyplot as plt

def g1stEQg0():

    eta = 0.0
    kVec = np.linspace(0, 2. * np.pi, prms.chainLength, endpoint=False)
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

def g1NumEQg1AnaGreater():
    eta = 1. / np.sqrt(prms.chainLength)
    kVec = np.linspace(0, 2. * np.pi, prms.chainLength, endpoint=False)
    tVec = np.linspace(0., 20. , 10)

    gfAna = greenAna.anaGreenVecTGreater(kVec, tVec, eta, 0.)
    gfNum = green.gfNumVecTGreater(kVec, tVec, eta, 0.)

    #fig = plt.figure()
    #ax = fig.add_subplot(111)
    #ax.plot(tVec, np.imag(gfAna[8, :]), color = 'red')
    #ax.plot(tVec, np.imag(gfNum[8, :]), color = 'blue')
    #plt.show()

    failArr = (np.abs(gfAna - gfNum) > 1e-5)
    print("Max difference of G-Greater = {}".format(np.amax(np.abs(gfAna - gfNum))))

    if(np.any(failArr)):
        print("Numerical GF1 Greater not consistent with analytical GF1!!! ------ CHECK FAILED!!!")
        return False
    else:
        print("Numerical GF1 Greater consistent with analytical GF1! ------ CHECK PASSED :)")
        return True

def g1NumEQg1AnaLesser():
    eta = 1. / np.sqrt(prms.chainLength)
    kVec = np.linspace(0, 2. * np.pi, prms.chainLength, endpoint=False)
    tVec = np.linspace(0., 10. , 10)

    gfAna = greenAna.anaGreenVecTLesser(kVec, tVec, eta, 0.)
    gfNum = green.gfNumVecTLesser(kVec, tVec, eta, 0.)

    failArr = (np.abs(gfAna - gfNum) > 1e-5)
    print("Max difference of G-Greater = {}".format(np.amax(np.abs(gfAna - gfNum))))

    if(np.any(failArr)):
        print("Numerical GF1 Lesser not consistent with analytical GF1!!! ------ CHECK FAILED!!!")
        return False
    else:
        print("Numerical GF1 Lesser consistent with analytical GF1! ------ CHECK PASSED :)")
        return True

def runAllTests():
    check1 = g1stEQg0()
    check2 = g1NumEQg1AnaGreater()
    check3 = g1NumEQg1AnaLesser()

    print("---------------------------")
    print("--- Green's function tests finished! ---")
    print("---------------------------")
    success = check1 and check2 and check3
    util.printSuccessMessage(success)
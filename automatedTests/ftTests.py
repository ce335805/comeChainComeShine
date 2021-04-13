import numpy as np
import matplotlib.pyplot as plt
import fourierTrafo as FT
from automatedTests import testUtils as util

def ftTest():

    w0 = 1.
    delta = .1
    tVec = np.linspace(-200., 200., 1000, endpoint=False)
    wVec = FT.wVecFromTVec(tVec)
    f = np.exp(1j * w0 * tVec - delta * np.abs(tVec))
    g = - 1. / np.sqrt(2. * np.pi) * (1j / (wVec - w0 - 1j * delta) - 1j / (wVec - w0 + 1j * delta))

    wVec, gFT = FT.FT(tVec, f)

    fig, ax = plt.subplots(nrows=1, ncols=1)
    #ax.plot(wVec, np.real(g), color = 'blue', label="Analytical")
    #ax.plot(wVec, np.real(gFT), color = 'red', label="Numerical")
    ax.plot(wVec, np.real(gFT) - np.real(g), color = 'red', label="Diff")
    #ax.plot(tVec, np.real(f), color = 'red', label="f(t)")
    plt.legend()
    plt.show()

    failArr = (np.abs(g - gFT) > 0.1)

    if(np.any(failArr)):
        print("Numerical FT is NOT consistent with analytical result!!! ------ CHECK FAILED!!!")
        return False
    else:
        print("Numerical FT is consistent with analytical result! ------ CHECK PASSED :)")
        return True


def ftTestPos():

    w0 = 40.
    delta = .5
    tVec = np.linspace(-200., 200., 30000, endpoint=False)
    wVec = FT.wVecFromTVec(tVec)
    f = np.exp(1j * w0 * tVec - delta * np.abs(tVec))
    tCut = np.zeros((len(tVec)))
    tCut[len(tVec)//2 : ] = 1.
    f = f * tCut
    #print(tVec * tCut)

    g = - 1. / np.sqrt(2. * np.pi) * (1j / (wVec - w0 - 1j * delta))

    wVec, gFT = FT.FT(tVec, f)

    #fig, ax = plt.subplots(nrows=1, ncols=1)
    #ax.plot(wVec, np.real(g), color = 'blue', label="Analytical")
    #ax.plot(wVec, np.real(gFT), color = 'red', label="Numerical")
    #ax.plot(tVec, np.real(f), color = 'red', label="f(t)")
    #plt.legend()
    #plt.show()

    failArr = (np.abs(g - gFT) > 0.1)

    if(np.any(failArr)):
        print("Numerical FT is NOT consistent with analytical result!!! ------ CHECK FAILED!!!")
        return False
    else:
        print("Numerical FT is consistent with analytical result! ------ CHECK PASSED :)")
        return True

def runAllTests():
    check1 = ftTest()
    check2 = ftTestPos()

    print("---------------------------")
    print("--- Fourier-Trafo tests finished! ---")
    print("---------------------------")
    success = check1 and check2
    util.printSuccessMessage(success)

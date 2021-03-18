import numpy as np
import greensFunction as green
import globalSystemParams as prms


def g1stEQg0():

    eta = 0.0
    kVec = np.linspace(0, 2. * np.pi, prms.chainLength)
    tVec = np.linspace(0., 10. , 10)
    gfT = green.anaGreenVecT(kVec, tVec, eta)
    g0T = green.g0VecT(kVec, tVec)

    failArr= (np.abs(gfT - g0T) > prms.accuracy)
    if(np.any(failArr)):
        print("G 1st not consistent with G0!!! ------ CHECK FAILED!!!")
        exit()
    else:
        print("G 1st consistent with G0! ------ CHECK PASSED :)")



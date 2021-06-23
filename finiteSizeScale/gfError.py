import numpy as np
import h5py

from greensFunction import greenAna1st
from greensFunction import greenAna2nd
from greensFunction import greenNumArb


import globalSystemParams as prms



def saveGfErrorForL(etaNonNorm, systemSize):
    eta = etaNonNorm / np.sqrt(systemSize)
    originalSize = prms.chainLength
    prms.chainLength = systemSize

    damping = 0.05
    kVec = np.linspace(0, 2. * np.pi, prms.chainLength, endpoint=False)
    wVec = np.linspace(-4, 4, 1000, endpoint=False)
    gfNumInf = greenNumArb.numGreenVecWGreater(kVec, wVec, eta, damping)
    gfAna2 = greenAna2nd.anaGreenVecW(kVec, wVec, eta, damping)
    gfAna1 = greenAna1st.anaGreenVecWGreater(kVec, wVec, eta, damping)
    print("gsNum1L.shape = {}".format(gfNumInf.shape))

    gfDiff1 = np.abs(np.imag(gfNumInf - gfAna1))
    gfDiff2 = np.abs(np.imag(gfNumInf - gfAna2))

    file = h5py.File("data/gfErr" + str(prms.chainLength) + ".h5", 'w')
    file.create_dataset("gfDiff1", data=gfDiff1)
    file.create_dataset("gfDiff2", data=gfDiff2)
    file.close()

    prms.chainLength = originalSize

def gfErrorForLs(etaNonNorm, Ls):

    for indL, L in enumerate(Ls):
        saveGfErrorForL(etaNonNorm, L)

def getMeanErrors(eta, Ls):

    means1 = np.zeros(Ls.shape)
    means2 = np.zeros(Ls.shape)
    for indL, L in enumerate(Ls):
        file = h5py.File("data/gfErr" + str(L) + ".h5", 'r')
        gfErrMean1 = np.mean(file["gfDiff1"][()])
        gfErrMean2 = np.mean(file["gfDiff2"][()])
        file.close()
        means1[indL] = gfErrMean1
        means2[indL] = gfErrMean2
    return [means1, means2]


def getMaxErrors(eta, Ls):
    means1 = np.zeros(Ls.shape)
    means2 = np.zeros(Ls.shape)
    for indL, L in enumerate(Ls):
        file = h5py.File("data/gfErr" + str(L) + ".h5", 'r')
        gfErrMean1 = np.amax(file["gfDiff1"][()])
        gfErrMean2 = np.amax(file["gfDiff2"][()])
        file.close()
        means1[indL] = gfErrMean1
        means2[indL] = gfErrMean2
    return [means1, means2]


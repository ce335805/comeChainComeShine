import numpy as np

import globalSystemParams as prms
from fourierTrafo import tVecFromWVec
import fourierTrafo as FT

def gLesser1st(kVec, tVec, A0):
    epsK = 2. * prms.t * np.cos(kVec)
    gK = - 2. * prms.t * np.sin(kVec)
    fac1 = np.exp(-1j * epsK[:, None] * tVec[None, :])
    fac2 = np.exp(-1j * A0 * gK[:, None] / prms.w0 * (1 - np.cos(prms.w0 * tVec[None, :])))
    return 1j * fac1 * fac2


def gLesserWithOcc(kVec, tVec, A0, damping):
        gs = np.zeros(prms.chainLength)
        gs[0 : prms.numberElectrons // 2 + 1] = 1.
        gs[- prms.numberElectrons // 2 + 1 :] = 1.
        print(gs)
        _, occupations = np.meshgrid(np.ones(tVec.shape), gs[:])
        GF = gLesser1st(kVec, tVec, A0)
        GF = np.multiply(occupations, GF)

        dampingArr, _ = np.meshgrid(np.exp(- damping * np.abs(tVec)), np.ones(kVec.shape))
        GF = np.multiply(dampingArr, GF)

        return GF

def gLesserW(kVec, wVec, A0, damping):
    tVec = FT.tVecFromWVec(wVec)
    #tVecNeg = tVec[: len(tVec) // 2 + 1]
    GFT = gLesserWithOcc(kVec, tVec, A0, damping)
    #GFZero = np.zeros((len(kVec), len(tVec) // 2 - 1), dtype='complex')
    #GFT = np.concatenate((GFT, GFZero), axis=1)
    wVecCheck, GFW = FT.FT(tVec, GFT)
    assert ((np.abs(wVec - wVecCheck) < 1e-10).all)
    return GFW

def spectralLesser(kVec, wVec, A0, damping):
    return -2. * np.imag(gLesserW(kVec, wVec, A0, damping))

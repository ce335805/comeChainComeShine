import numpy as np

import globalSystemParams as prms
import fourierTrafo as FT

def gLesser(kVec, tRelArr, tAvArr, A0):

    gRet = np.zeros((len(kVec), len(tRelArr), len(tAvArr)), dtype=complex)
    for tAvInd, tAv in enumerate(tAvArr):
        for tRelInd, tRel in enumerate(tRelArr):
            t = tAv + .5 * tRel
            tDash = tAv - .5 * tRel
            if(t > tDash):
                continue
            dHdt = integrateH(np.linspace(0., t, 1000, endpoint=True), A0)
            dHdtDash = integrateH(np.linspace(0., tDash, 1000, endpoint=True), A0)
            dHdtdtDash = integrateHPlus(kVec, np.linspace(tDash, t, 1000, endpoint=True), A0)
            gRet[:, tRelInd, tAvInd] = np.exp(1j * dHdt) * np.exp(-1j * dHdtDash) * np.exp(-1j * dHdtdtDash)
    return 1j * gRet

def integrateH(tVec, A0):
    At = A0 * np.sin(prms.w0 * tVec)
    T = - 4. * prms.t
    H = np.cos(At) * T
    return (tVec[1] - tVec[0]) * np.sum(H[: -1])

def integrateHPlus(kVec, tVec, A0):
    At = A0 * np.sin(prms.w0 * tVec)
    T = - 4. * prms.t
    H = np.cos(At) * T
    epsK = 2. * prms.t * np.cos(kVec)
    gK = - 2. * prms.t * np.sin(kVec)
    Hext = H[None, :] - np.cos(At)[None, :] * epsK[:, None] - np.sin(At)[None, :] * gK[:, None]
    return (tVec[1] - tVec[0]) * np.sum(Hext[:, :-1], axis=1)


def gLesserWithOcc(kVec, tRelArr, tAvArr, A0, damping):
        gs = np.zeros(prms.chainLength)
        gs[0 : prms.numberElectrons // 2 + 1] = 1.
        gs[- prms.numberElectrons // 2 + 1 :] = 1.
        tRelOnes = np.ones(tRelArr.shape)
        tAvOnes = np.ones(tAvArr.shape)
        kOnes = np.ones(kVec.shape)
        occupations = gs[:, None, None] * tRelOnes[None, :, None] * tAvOnes[None, None, :]
        GF = gLesser(kVec, tRelArr, tAvArr, A0)
        GF = np.multiply(occupations, GF)

        dampingArr = np.exp(- damping * np.abs(tRelArr))[None, :, None] * kOnes[:, None, None] * tAvOnes[None, None, :]
        GF = np.multiply(dampingArr, GF)

        return GF

def gLesserW(kVec, wRelVec, tAvArr, A0, damping):
    tVec = FT.tVecFromWVec(wRelVec)
    tVecNeg = tVec[: len(tVec) // 2 + 1]
    GFT = gLesserWithOcc(kVec, tVecNeg, tAvArr, A0, damping)
    GFZero = np.zeros((len(kVec), len(tVec) // 2 - 1, len(tAvArr)), dtype='complex')
    GFT = np.concatenate((GFT, GFZero), axis=1)
    wVecCheck, GFW = FT.FTOneOfTwoTimes(tVec, GFT)
    assert ((np.abs(wRelVec - wVecCheck) < 1e-10).all)
    return GFW

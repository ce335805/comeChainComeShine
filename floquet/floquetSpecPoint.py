import numpy as np

import globalSystemParams as prms
import fourierTrafo as FT
from coherentState import coherentState


def gLesser(kPoint, tRelArr, tAvArr, eta, N):
    A0 = getA0Coh(eta, N)
    print("A0 = {}".format(A0))

    gRet = np.zeros((len(tRelArr), len(tAvArr)), dtype=complex)
    for tAvInd, tAv in enumerate(tAvArr):
        for tRelInd, tRel in enumerate(tRelArr):
            t = tAv + .5 * tRel
            tDash = tAv - .5 * tRel
            if (t > tDash):
                continue
            dHdt = -integrateH(np.linspace(0., t, 1000, endpoint=True), A0)
            dHdtDash = -integrateH(np.linspace(0., tDash, 1000, endpoint=True), A0)
            dHdtdtDash = -integrateHPlus(kPoint, np.linspace(tDash, t, 1000, endpoint=True), A0)
            gRet[tRelInd, tAvInd] = np.exp(1j * dHdt) * np.exp(-1j * dHdtDash) * np.exp(-1j * dHdtdtDash)
    return 1j * gRet


def integrateH(tVec, A0):
    At = A0 * np.sin(prms.w0 * tVec)
    T = 4. * prms.t * prms.numberElectrons
    H = np.cos(At) * T
    return (tVec[1] - tVec[0]) * np.sum(H[: -1])


def integrateHPlus(kPoint, tVec, A0):
    At = A0 * np.sin(prms.w0 * tVec)
    T = 4. * prms.t * prms.numberElectrons
    H = np.cos(At) * T
    epsK = 2. * prms.t * np.cos(kPoint)
    gK = - 2. * prms.t * np.sin(kPoint)
    Hext = H[:] - np.cos(At)[:] * epsK + np.sin(At)[:] * gK
    return (tVec[1] - tVec[0]) * np.sum(Hext[:-1], axis=0)


def getA0Coh(eta, N):
    phState = coherentState.getCoherentStateForN(N)
    apad = np.diag(np.sqrt(np.arange((prms.maxPhotonNumber - 1)) + 1), -1) + np.diag(
        np.sqrt(np.arange((prms.maxPhotonNumber - 1)) + 1), +1)
    aAv = np.dot(np.conj(phState), np.dot(eta * apad, phState))
    return aAv


def gLesserDamp(kPoint, tRelArr, tAvArr, eta, damping, cohN):

    tAvOnes = np.ones(tAvArr.shape)
    GF = gLesser(kPoint, tRelArr, tAvArr, eta, cohN)
    dampingArr = np.exp(- damping * np.abs(tRelArr))[:, None] * tAvOnes[None, :]
    GF = np.multiply(dampingArr, GF)

    return GF


def gLesserW(kPoint, wRelVec, tAvArr, eta, damping, cohN):
    tVec = FT.tVecFromWVec(wRelVec)
    tVecNeg = tVec[: len(tVec) // 2 + 1]
    GFT = gLesserDamp(kPoint, tVecNeg, tAvArr, eta, damping, cohN)
    GFZero = np.zeros((len(tVec) // 2 - 1, len(tAvArr)), dtype='complex')
    GFT = np.concatenate((GFT, GFZero), axis=0)
    wVecCheck, GFW = FT.FTOneOfTwoTimesPoint(tVec, GFT)
    assert ((np.abs(wRelVec - wVecCheck) < 1e-10).all)
    return GFW

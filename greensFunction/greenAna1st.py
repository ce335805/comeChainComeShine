import numpy as np
import sec_order.analyticalEGS as anaGS
import globalSystemParams as prms
import energyFunctions as eF
from arb_order import photonState as phState
from arb_order import numHamiltonians as numH
import scipy.linalg as sciLin
import fourierTrafo as FT


def anaGreenPointTGreater(kPoint, tPoint, gsJ, eta):
    epsK = 2. * prms.t * np.cos(kPoint[:, None])
    coupling = - eta ** 2 / prms.w0 * \
               (2. * ( gsJ * (-2.) * prms.t * np.sin(kPoint[:, None])) + (-2. * prms.t * np.sin(kPoint[:, None]))**2)

    coupling = - eta ** 2 / prms.w0 * (-2. * prms.t * np.sin(kPoint[:, None])**2)


    eTime = -1j * epsK * tPoint[None, :] - 1j * coupling * tPoint[None, :]
    ptTime = - (- 2. * eta * prms.t * np.sin(kPoint[:, None]))**2 / prms.w0**2 * (1. - np.exp(-1j * prms.w0 * tPoint[None, :]))
    return -1j * np.exp(eTime + ptTime)

def anaGreenPointTLesser(kPoint, tPoint, gsJ, eta):
    epsK = 2. * prms.t * np.cos(kPoint[:, None])
    coupling = - eta ** 2 / prms.w0 * \
               (2. * ( gsJ * (-2.) * prms.t * np.sin(kPoint[:, None])) + (-2. * prms.t * np.sin(kPoint[:, None]))**2)

    eTime = -1j * epsK * tPoint[None, :] - 1j * coupling * tPoint[None, :]
    ptTime = - (- 2. * eta * prms.t * np.sin(kPoint[:, None]))**2 / prms.w0**2 * (1. - np.exp(-1j * prms.w0 * tPoint[None, :]))
    return 1j * np.exp(eTime + ptTime)


def anaGreenVecTGreater(kVec, tVec, eta, damping):

    gs = anaGS.findGS1st(eta)
    gsJ = eF.J(gs[0: -1])
    #_, occupations = np.meshgrid(np.ones(tVec.shape), gs[0: -1])
    GF = anaGreenPointTGreater(kVec, tVec, gsJ, eta)
    #GF = np.multiply(1 - occupations, GF)

    dampingArr, _ = np.meshgrid(np.exp(- damping * np.abs(tVec)), np.ones(kVec.shape))
    GF = np.multiply(dampingArr, GF)

    return GF

def anaGreenVecTLesser(kVec, tVec, eta, damping):

    gs = anaGS.findGS1st(eta)
    gsJ = eF.J(gs[0: -1])
    _, occupations = np.meshgrid(np.ones(tVec.shape), gs[0: -1])
    GF = anaGreenPointTLesser(kVec, tVec, gsJ, eta)
    GF = np.multiply(occupations, GF)

    dampingArr, _ = np.meshgrid(np.exp(- damping * np.abs(tVec)), np.ones(kVec.shape))
    GF = np.multiply(dampingArr, GF)

    return GF


def anaGreenVecWGreater(kVec, wVec, eta, damping):
    tVec = FT.tVecFromWVec(wVec)
    tVecPos = tVec[len(tVec) // 2 : ]
    GFT = anaGreenVecTGreater(kVec, tVecPos, eta, damping)
    GFZero = np.zeros((len(kVec), len(tVec)//2), dtype='complex')
    GFT = np.concatenate((GFZero, GFT), axis=1)

    wVecCheck, GFW = FT.FT(tVec, GFT)

    assert((np.abs(wVec - wVecCheck) < 1e-10).all)
    return GFW

def anaGreenVecWLesser(kVec, wVec, eta, damping):
    tVec = FT.tVecFromWVec(wVec)
    tVecNeg = tVec[: len(tVec) // 2 + 1]
    GFT = anaGreenVecTLesser(kVec, tVecNeg, eta, damping)
    GFZero = np.zeros((len(kVec), len(tVec)//2), dtype='complex')
    GFT = np.concatenate((GFT, GFZero), axis=1)

    wVecCheck, GFW = FT.FT(tVec, GFT)

    assert((np.abs(wVec - wVecCheck) < 1e-10).all)
    return GFW

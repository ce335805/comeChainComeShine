import numpy as np
import sec_order.analyticalEGS as anaGS
import globalSystemParams as prms
import energyFunctions as eF
from arb_order import photonState as phState
from arb_order import numHamiltonians as numH
import scipy.linalg as sciLin
import fourierTrafo as FT
from arb_order import arbOrder


def anaGreenPointT(kPoint, tPoint, gsJ, eta):

    gs = arbOrder.findGS(eta, 3)
    gsJ = eF.J(gs)
    gsT = eF.T(gs)

    #gsT = - 2. / np.pi * prms.chainLength
    z = 2. * eta**2 / prms.w0 * gsT
    wTilde = prms.w0 * np.sqrt(1. - z)

    epsK = 2. * prms.t * np.cos(kPoint[:, None]) * (1. - 0.5 * eta**2 * prms.w0 / wTilde)
    selfE = - eta**2 * prms.w0 / (wTilde**2) * (- 2. * prms.t * np.sin(kPoint[:, None]))**2

    eTime = -1j * epsK * tPoint[None, :] - 1j * selfE * tPoint[None, :]
    ptTime = - (- 2. * eta * prms.t * np.sin(kPoint[:, None]))**2 * prms.w0 / (wTilde**3) * (1. - np.exp(-1j * wTilde * tPoint[None, :]))

    return -1j * np.exp(eTime + ptTime)

def anaGreenPointTLesser(kPoint, tPoint, gsJ, eta):

    gs = arbOrder.findGS(eta, 3)
    gsJ = eF.J(gs)
    gsT = eF.T(gs)

    #gsT = - 2. / np.pi * prms.chainLength
    z = 2. * eta**2 / prms.w0 * gsT
    wTilde = prms.w0 * np.sqrt(1. - z)

    epsK = 2. * prms.t * np.cos(kPoint[:, None]) * (1. - 0.5 * eta**2 * prms.w0 / wTilde)
    selfE = eta**2 * prms.w0 / (wTilde**2) * (- 2. * prms.t * np.sin(kPoint[:, None]))**2

    eTime = -1j * epsK * tPoint[None, :] - 1j * selfE * tPoint[None, :]
    ptTime = - (- 2. * eta * prms.t * np.sin(kPoint[:, None]))**2 * prms.w0 / (wTilde**3) * (1. - np.exp(1j * wTilde * tPoint[None, :]))

    return -1j * np.exp(eTime + ptTime)


def anaGreenVecT(kVec, tVec, eta, damping):
    if(len(kVec) == prms.chainLength):
        gs = arbOrder.findGS(eta, 3)
    else:#don't include occupations if length of kVec doesnt match chain length
        gs = np.ones(len(kVec), dtype=complex)
    _, occupations = np.meshgrid(np.ones(tVec.shape), gs[:])
    gsJ = 0.
    GF = anaGreenPointT(kVec, tVec, gsJ, eta)
    GF = np.multiply(1 - occupations, GF)
    dampingArr, _ = np.meshgrid(np.exp(- damping * np.abs(tVec)), np.ones(kVec.shape))
    GF = np.multiply(dampingArr, GF)

    return GF

def anaGreenVecTLesser(kVec, tVec, eta, damping):
    if(len(kVec) == prms.chainLength):
        gs = arbOrder.findGS(eta, 3)
    else:
        gs = np.ones(kVec.shape, dtype=complex)
    _, occupations = np.meshgrid(np.ones(tVec.shape), gs[:])
    gsJ = 0.
    GF = anaGreenPointTLesser(kVec, tVec, gsJ, eta)
    GF = np.multiply(occupations, GF)
    dampingArr, _ = np.meshgrid(np.exp(- damping * np.abs(tVec)), np.ones(kVec.shape))
    GF = np.multiply(dampingArr, GF)

    return GF


def anaGreenVecW(kVec, wVec, eta, damping):
    tVec = FT.tVecFromWVec(wVec)
    tVecPos = tVec[len(tVec) // 2: ]
    GFT = anaGreenVecT(kVec, tVecPos, eta, damping)
    GFZero = np.zeros((len(kVec), len(tVec)//2), dtype='complex')
    GFT = np.concatenate((GFZero, GFT), axis=1)

    wVecCheck, GFW = FT.FT(tVec, GFT)

    assert((np.abs(wVec - wVecCheck) < 1e-10).all)
    return GFW

def anaGreenVecWLesser(kVec, wVec, eta, damping):
    tVec = FT.tVecFromWVec(wVec)
    tVecPos = tVec[len(tVec) // 2: ]
    GFT = anaGreenVecTLesser(kVec, tVecPos, eta, damping)
    GFZero = np.zeros((len(kVec), len(tVec)//2), dtype='complex')
    GFT = np.concatenate((GFZero, GFT), axis=1)

    wVecCheck, GFW = FT.FT(tVec, GFT)

    assert((np.abs(wVec - wVecCheck) < 1e-10).all)
    return GFW


import numpy as np
import sec_order.analyticalEGS as anaGS
import globalSystemParams as prms
import energyFunctions as eF
from arb_order import photonState as phState
from arb_order import numHamiltonians as numH
import scipy.linalg as sciLin
import fourierTrafo as FT


def anaGreenPointT(kPoint, tPoint, gsJ, eta):

    gsT = - 2. / np.pi * prms.chainLength
    z = 2. * eta**2 / prms.w0 * gsT
    wTilde = prms.w0 * np.sqrt(1. - z)
    wDash = prms.w0 * (1. - z)



    epsK = - 2. * prms.t * np.cos(kPoint[:, None]) * (1. - 0.5 * eta**2 * wTilde / prms.w0)
    fac = .5 * (2. - z) / (1. - z)
    selfE = - eta**2 / (wTilde) * np.sqrt(np.sqrt(1 - z)) * (- 2. * prms.t * np.sin(kPoint[:, None]))**2

    eTime = -1j * epsK * tPoint[None, :] - 1j * selfE * tPoint[None, :]

    ptTime = np.zeros((len(kPoint), len(tPoint)), dtype=complex)
    for indK, k in enumerate(kPoint):
        if(k > -np.pi / 2. and k <= np.pi / 2):
            ptTimeTemp = - (- 2. * eta * prms.t * np.sin(k))**2 / (wTilde * prms.w0) * (1. - np.exp(- 1j * wTilde * tPoint[:]))
            ptTime[indK, :] = ptTimeTemp
        else:
            ptTimeTemp = - (- 2. * eta * prms.t * np.sin(k)) ** 2 / (wTilde * prms.w0) * (1. - np.exp( 1j * wTilde * tPoint[:]))
            ptTime[indK, :] = ptTimeTemp

    return -1j * np.exp(eTime + ptTime)


def anaGreenVecT(kVec, tVec, eta, damping):

    gsJ = 0.
    GF = anaGreenPointT(kVec, tVec, gsJ, eta)
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


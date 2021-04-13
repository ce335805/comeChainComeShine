import numpy as np
import sec_order.analyticalEGS as anaGS
import globalSystemParams as prms
import energyFunctions as eF
from arb_order import photonState as phState
from arb_order import numHamiltonians as numH
import scipy.linalg as sciLin
import fourierTrafo as FT
from arb_order import arbOrder
from coherentState import coherentState


def gfPointTCoh(kVec, tAv, tRel, eta):
    phState = coherentState.getCoherentStateForN(1.)
    H = getH(eta)

    gk = - 2. * prms.t * np.sin(kVec)
    eK = 2. * prms.t * np.cos(kVec)
    x = eta * (np.diag(np.sqrt(np.arange((prms.maxPhotonNumber - 1)) + 1), -1) + np.diag(
        np.sqrt(np.arange((prms.maxPhotonNumber - 1)) + 1), +1))
    sinX = sciLin.sinm(x)
    cosX = sciLin.cosm(x)

    iHTRelMTAv, iHTRelPTAv, iHtSinCosTRel = setUpMatricies(H, cosX, eK, gk, sinX, tAv, tRel)

    expiHt, expiHtDash, exptRel = setUpExponentialMatricies(iHTRelMTAv, iHTRelPTAv, iHtSinCosTRel, kVec, tAv, tRel)

    GF = evaluateBraKet(expiHt, expiHtDash, exptRel, kVec, phState, tAv, tRel)

    return - 1j * GF


def gfNumPointTGS(kVec, tAv, tRel, eta):
    phState = getPhGS(eta)
    H = getH(eta)

    gk = - 2. * prms.t * np.sin(kVec)
    eK = 2. * prms.t * np.cos(kVec)
    x = eta * (np.diag(np.sqrt(np.arange((prms.maxPhotonNumber - 1)) + 1), -1) + np.diag(
        np.sqrt(np.arange((prms.maxPhotonNumber - 1)) + 1), +1))
    sinX = sciLin.sinm(x)
    cosX = sciLin.cosm(x)

    iHTRelMTAv, iHTRelPTAv, iHtSinCosTRel = setUpMatricies(H, cosX, eK, gk, sinX, tAv, tRel)

    expiHt, expiHtDash, exptRel = setUpExponentialMatricies(iHTRelMTAv, iHTRelPTAv, iHtSinCosTRel, kVec, tAv, tRel)

    GF = evaluateBraKet(expiHt, expiHtDash, exptRel, kVec, phState, tAv, tRel)

    return - 1j * GF


def evaluateBraKet(expiHt, expiHtDash, exptRel, kVec, phState, tAv, tRel):
    prod1 = np.zeros((len(tRel), len(tAv), prms.maxPhotonNumber), dtype='complex')
    print("prod1.shape = {}".format(prod1.shape))
    for tAvInd in range(len(tAv)):
        for tRelInd in range(len(tRel)):
            prod1[tAvInd, tRelInd, :] = np.dot(expiHtDash[tAvInd, tRelInd, :, :], phState[:])
    prod2 = np.zeros((len(kVec), len(tRel), len(tAv), prms.maxPhotonNumber), dtype='complex')
    for tAvInd in range(len(tAv)):
        for tRelInd in range(len(tRel)):
            for kInd in range(len(kVec)):
                prod2[kInd, tRelInd, tAvInd, :] = np.dot(exptRel[kInd, tRelInd, :, :], prod1[tRelInd, tAvInd, :])
    prod3 = np.zeros((len(kVec), len(tRel), len(tAv), prms.maxPhotonNumber), dtype='complex')
    for tAvInd in range(len(tAv)):
        for tRelInd in range(len(tRel)):
            for kInd in range(len(kVec)):
                prod3[kInd, tRelInd, tAvInd, :] = np.dot(expiHt[tRelInd, tAvInd, :, :], prod2[kInd, tRelInd, tAvInd, :])
    GF = np.zeros((len(kVec), len(tRel), len(tAv)), dtype='complex')
    for tAvInd in range(len(tAv)):
        for tRelInd in range(len(tRel)):
            for kInd in range(len(kVec)):
                GF[kInd, tRelInd, tAvInd] = np.dot(np.conj(phState[:]), prod3[kInd, tRelInd, tAvInd, :])
    return GF


def setUpExponentialMatricies(iHTRelMTAv, iHTRelPTAv, iHtSinCosTRel, kVec, tAv, tRel):
    exptRel = np.zeros((len(kVec), len(tRel), prms.maxPhotonNumber, prms.maxPhotonNumber), dtype=complex)
    for tRelInd in range(len(tRel)):
        for kInd in range(len(kVec)):
            exptRel[kInd, tRelInd] = sciLin.expm(iHtSinCosTRel[kInd, tRelInd, :, :])
    expiHt = np.zeros((len(tRel), len(tAv), prms.maxPhotonNumber, prms.maxPhotonNumber), dtype=complex)
    expiHtDash = np.zeros((len(tRel), len(tAv), prms.maxPhotonNumber, prms.maxPhotonNumber), dtype=complex)
    for tAvInd in range(len(tAv)):
        for tRelInd in range(len(tRel)):
            expiHt[tRelInd, tAvInd, :, :] = sciLin.expm(iHTRelPTAv[tRelInd, tAvInd, :, :])
            expiHtDash[tRelInd, tAvInd, :, :] = sciLin.expm(iHTRelMTAv[tRelInd, tAvInd, :, :])
    return expiHt, expiHtDash, exptRel


def setUpMatricies(H, cosX, eK, gk, sinX, tAv, tRel):
    iHtSinCosTRel = - 1j * H[None, None, :, :] * tRel[None, :, None, None] \
                    - 1j * gk[:, None, None, None] * sinX[None, None, :, :] * tRel[None, :, None, None] \
                    - 1j * eK[:, None, None, None] * cosX[None, None, :, :] * tRel[None, :, None, None]
    iHTRelMTAv = - .5j * H[None, None, :, :] * (tAv[None, :, None, None] - tRel[:, None, None, None])
    iHTRelPTAv = .5j * H[None, None, :, :] * (tAv[None, :, None, None] + tRel[:, None, None, None])
    return iHTRelMTAv, iHTRelPTAv, iHtSinCosTRel


def getPhGS(eta):
    gs = arbOrder.findGS(eta, 3)
    gsJ = eF.J(gs)
    gsT = eF.T(gs)
    return phState.findPhotonGS([gsT, gsJ], eta, 3)


def getH(eta):
    gs = arbOrder.findGS(eta, 3)
    gsJ = eF.J(gs)
    gsT = eF.T(gs)
    return numH.setupPhotonHamiltonianInf(gsT, gsJ, eta)

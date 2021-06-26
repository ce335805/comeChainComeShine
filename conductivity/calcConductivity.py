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
import matplotlib.pyplot as plt

def calcConductivityAna(omegaVec, delta, eta):
    gsT = - 2. / np.pi * prms.chainLength

    fac = np.sqrt(1 - 2. * eta * eta / (prms.w0) * gsT)

    drudePart = -1j / (omegaVec + 1j * delta) * gsT / prms.chainLength * (1 - eta ** 2 / (2. * fac) )

    currentcurrentPart = - 1j * eta**2 / fac * (1j / (omegaVec - fac * prms.w0 + 1j * delta) - 1j / (omegaVec + fac * prms.w0 + 1j * delta))

    cavityPart = gsT ** 2 / prms.chainLength * 1j / (omegaVec + 1j * delta) * currentcurrentPart
    #cavityPart = gsT ** 2 / prms.chainLength * currentcurrentPart

    cond = drudePart + cavityPart
    return cond

def calcConductivityNum(omegaVec, delta, eta):
    gsT = - 2. / np.pi * prms.chainLength

    eKinSupp = expectationCos(eta)
    #eKinSuppAna = 1. - eta**2 /(2. * np.sqrt(1. - 2. * eta**2 / prms.w0 * gsT))

    drudePart = -1j / (omegaVec + 1j * delta) * gsT / prms.chainLength * eKinSupp

    currentcurrentPart = expectationSinSinW(omegaVec, eta, delta)
    #need extra '-' at omega since I calculate correlator as function of -w
    #extra conj also because of negative frequencies
    cavityPart = np.conj(gsT ** 2 / prms.chainLength * 1j / (-omegaVec + 1j * delta) * currentcurrentPart)
    cond = drudePart + cavityPart
    return cond

def expectationCos(eta):

    phGS = getPhGS(eta)
    #phGS = coherentState.getSqueezedState(eta, gsT)
    x = eta * (np.diag(np.sqrt(np.arange((prms.maxPhotonNumber - 1)) + 1), -1) + np.diag(
        np.sqrt(np.arange((prms.maxPhotonNumber - 1)) + 1), +1))
    cosX = sciLin.cosm(x)

    cosExpectation = np.dot(np.transpose(np.conj(phGS)), np.dot(cosX, phGS))

    return cosExpectation

def gsEffectiveKineticEnergyArrayNum(etaArr):
    gsT = - 2./np.pi * prms.chainLength
    gsKinetics = np.zeros(len(etaArr))
    for etaInd, eta in enumerate(etaArr):
        gsKinetics[etaInd] = np.real(expectationCos(eta)) * gsT / prms.chainLength
    return gsKinetics


def expectationSinSin(tVec, eta):

    phGS = getPhGS(eta)
    H = getH(eta)

    x = eta * (np.diag(np.sqrt(np.arange((prms.maxPhotonNumber - 1)) + 1), -1) + np.diag(
        np.sqrt(np.arange((prms.maxPhotonNumber - 1)) + 1), +1))
    sinX = sciLin.sinm(x)
    hEvals, hEvecs = np.linalg.eigh(H)

    expectationSin = np.zeros(tVec.shape, dtype='complex')
    for tInd, t in enumerate(tVec):
        expiHt = np.dot(hEvecs, np.dot(np.diag(np.exp(1j * hEvals * t)), np.transpose(np.conj(hEvecs))))
        prod1 = np.dot(sinX, phGS)
        prod2 = np.dot( np.conj(expiHt), prod1)
        prod3 = np.dot( sinX, prod2)
        prod4 = np.dot( expiHt, prod3)
        res = np.dot( np.conj(phGS), prod4)
        expectationSin[tInd] = res

    gsT = - 2. / np.pi * prms.chainLength
    fac = np.sqrt(1 - 2. * eta * eta / (prms.w0) * gsT)
    expectationSin = eta**2 / fac * np.exp(-1j * prms.w0 * fac * tVec)

    return 1j * expectationSin

def expectationSinSinTurned(tVec, eta):

    phGS = getPhGS(eta)
    H = getH(eta)

    x = eta * (np.diag(np.sqrt(np.arange((prms.maxPhotonNumber - 1)) + 1), -1) + np.diag(
        np.sqrt(np.arange((prms.maxPhotonNumber - 1)) + 1), +1))
    sinX = sciLin.sinm(x)
    hEvals, hEvecs = np.linalg.eigh(H)

    expectationSin = np.zeros(tVec.shape, dtype='complex')
    for tInd, t in enumerate(tVec):
        expiHt = np.dot(hEvecs, np.dot(np.diag(np.exp(1j * hEvals * t)), np.transpose(np.conj(hEvecs))))
        prod1 = np.dot(np.conj(expiHt), phGS)
        prod2 = np.dot( sinX, prod1)
        prod3 = np.dot( expiHt, prod2)
        prod4 = np.dot( sinX, prod3)
        res = np.dot( np.conj(phGS), prod4)
        expectationSin[tInd] = res

    gsT = - 2. / np.pi * prms.chainLength
    fac = np.sqrt(1 - 2. * eta * eta / (prms.w0) * gsT)
    expectationSin = eta**2 / fac * np.exp(1j * prms.w0 * fac * tVec)

    return 1j * expectationSin


def expectationSinSinW(wVec, eta, damping):
    tVec = FT.tVecFromWVec(wVec)
    tVecPos = tVec[len(tVec) // 2 : ]
    expSinSinPos = expectationSinSin(tVecPos, eta)
    expSinSinPosTurned = expectationSinSinTurned(tVecPos, eta)
    expSinSinPos = - expSinSinPos + expSinSinPosTurned
    Zeros = np.zeros((len(tVec)//2), dtype='complex')
    #expSinSin = np.concatenate((expSinSinPos, Zeros), axis=0)
    expSinSin = np.concatenate((Zeros, expSinSinPos), axis=0)
    dampingArr = np.exp(- damping *  np.abs(tVec))
    expSinSinDamped = expSinSin * dampingArr

    wVecCheck, expSinSinW = FT.FT(tVec, expSinSinDamped)

    #gsT = - 2. / np.pi * prms.chainLength
    #fac = np.sqrt(1 - 2. * eta * eta / (prms.w0) * gsT)
    #return eta**2 / fac * (1j / (wVec - fac * prms.w0 + 1j * damping) - 1j / (wVec + fac * prms.w0 + 1j * damping))

    return expSinSinW * np.sqrt(2. * np.pi)


def getPhGS(eta):
    gs = arbOrder.findGS(eta, 3)
    gsJ = eF.J(gs)
    gsT = eF.T(gs)
    gsJ = 0.
    gsT = - 2. / np.pi * prms.chainLength

    return phState.findPhotonGS([gsT, gsJ], eta, 3)


def getH(eta):
    gs = arbOrder.findGS(eta, 3)
    gsJ = eF.J(gs)
    gsT = eF.T(gs)
    return numH.setupPhotonHamiltonianInf(gsT, gsJ, eta)


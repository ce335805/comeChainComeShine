from multiprocessing import Pool
from functools import partial
import numpy as np

from nonEqGreen import nonEqGreenPoint
from arb_order import arbOrder
import energyFunctions as eF
import globalSystemParams as prms
from nonEqGreen import nonEqGreenPoint

import time

def nonEqGreenMultiParamMultiProc(kPoint, wVec, damping, nArr):

    startTime = time.time()

    gfNonEqPartial = partial(nonEqGFParameters, kPoint = kPoint, wVec = wVec, damping = damping)
    #gfNonEqCohOnlyKArgTurned = partial(nonEqGreenPoint.gfCohWTurned, wRel = wVec, tAv = tAv, eta = eta, damping = damping, N = cohN)

    pool = Pool()

    gf = pool.map(gfNonEqPartial, nArr)
    gf = np.array(gf)
    print("gf.shape = {}".format(gf.shape))

    pool.close()
    pool.join()

    stopTime = time.time()
    print("parallel execution non-eq green took {:.5f}".format(stopTime - startTime))



    return gf

def nonEqGFParameters(nCoh, kPoint, wVec, damping):

    if(nCoh == 0):
        eta = 2.5 / np.sqrt(prms.chainLength)
    else:
        eta = np.sqrt(0.4 / nCoh) / np.sqrt(prms.chainLength)

    print("nCoh = {} ; eta = {}".format(nCoh, eta * np.sqrt(prms.chainLength)))

    gs = arbOrder.findGS(eta, 3)
    gsT = eF.T(gs)
    wTilde = np.sqrt(1. - 2. * eta**2 / prms.w0 * gsT)
    tau = 2. * np.pi / wTilde
    tauLength = 1.
    tAv = np.linspace(1000. * tau, (1000. + tauLength) * tau, 20, endpoint=False)
    gfNonEq = nonEqGreenPoint.gfCohWTurned(kPoint = kPoint,wRel = wVec, tAv = tAv, eta = eta, damping = damping, N = nCoh)
    gfNonEqN0 = 1. / (tauLength * tau) * (tAv[1] - tAv[0]) * np.sum(gfNonEq, axis=1)

    return gfNonEqN0

def nonEqGreenMultiParamMultiProcAboveFS(kPoint, wVec, damping, nArr):

    startTime = time.time()

    gfNonEqPartial = partial(nonEqGFParametersAboveFS, kPoint = kPoint, wVec = wVec, damping = damping)
    #gfNonEqCohOnlyKArgTurned = partial(nonEqGreenPoint.gfCohWTurned, wRel = wVec, tAv = tAv, eta = eta, damping = damping, N = cohN)

    pool = Pool()

    gf = pool.map(gfNonEqPartial, nArr)
    gf = np.array(gf)
    print("gf.shape = {}".format(gf.shape))

    pool.close()
    pool.join()

    stopTime = time.time()
    print("parallel execution non-eq green took {:.5f}".format(stopTime - startTime))



    return gf

def nonEqGFParametersAboveFS(nCoh, kPoint, wVec, damping):

    if(nCoh == 0):
        eta = 2.5 / np.sqrt(prms.chainLength)
    else:
        eta = np.sqrt(0.4 / nCoh) / np.sqrt(prms.chainLength)

    print("nCoh = {} ; eta = {}".format(nCoh, eta * np.sqrt(prms.chainLength)))

    gs = arbOrder.findGS(eta, 3)
    gsT = eF.T(gs)
    wTilde = np.sqrt(1. - 2. * eta**2 / prms.w0 * gsT)
    tau = 2. * np.pi / wTilde
    tauLength = 1.
    tAv = np.linspace(1000. * tau, (1000. + tauLength) * tau, 20, endpoint=False)
    gfNonEq = nonEqGreenPoint.gfCohW(kPoint = kPoint,wRel = wVec, tAv = tAv, eta = eta, damping = damping, N = nCoh)
    gfNonEqN0 = 1. / (tauLength * tau) * (tAv[1] - tAv[0]) * np.sum(gfNonEq, axis=1)

    return gfNonEqN0

from multiprocessing import Pool
from functools import partial
import numpy as np

from nonEqGreen import nonEqGreenPoint

import time

def nonEqGreenMultiProc(kVec, wVec, tAv, eta, damping, cohN):

    startTime = time.time()

    gfNonEqCohOnlyKArg = partial(nonEqGreenPoint.gfCohW, wRel = wVec, tAv = tAv, eta = eta, damping = damping, N = cohN)
    gfNonEqCohOnlyKArgTurned = partial(nonEqGreenPoint.gfCohWTurned, wRel = wVec, tAv = tAv, eta = eta, damping = damping, N = cohN)

    pool = Pool()

    gf = pool.map(gfNonEqCohOnlyKArg, kVec)
    gfTurned = pool.map(gfNonEqCohOnlyKArgTurned, kVec)
    gf = np.array(gf) + np.array(gfTurned)
    #gf = np.array([nonEqGreenPoint.gfCohWLesser(k, wRel = wVec, tAv = tAv, eta = eta, damping = damping, N = cohN) for k in kVec])
    #print("gf.shape = {}".format(gf.shape))

    pool.close()
    pool.join()

    stopTime = time.time()
    print("parallel execution non-eq green took {:.5f}".format(stopTime - startTime))

    return gf

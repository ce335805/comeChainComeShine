from multiprocessing import Pool
from functools import partial

from nonEqGreen import nonEqGreenPoint

import time

def nonEqGreenMultiProc(kVec, wVec, tAv, eta, damping, cohN):

    startTime = time.time()

    gfNonEqCohOnlyKArg = partial(nonEqGreenPoint.gfCohWLesser, wRel = wVec, tAv = tAv, eta = eta, damping = damping, N = cohN)

    pool = Pool()

    gf = pool.map(gfNonEqCohOnlyKArg, kVec)

    pool.close()
    pool.join()

    stopTime = time.time()
    print("parallel execution non-eq green took {:.5f}".format(stopTime - startTime))

    return gf

from multiprocessing import Pool
from functools import partial

from floquet import floquetSpecPoint

import time

def floquetGreenMultiProc(kVec, wVec, tAv, eta, damping, cohN):

    startTime = time.time()

    floquetSpecPointKVec = partial(floquetSpecPoint.gLesserW, wRelVec = wVec, tAvArr = tAv, eta = eta, damping = damping, cohN = cohN)

    pool = Pool()

    gf = pool.map(floquetSpecPointKVec, kVec)

    pool.close()
    pool.join()

    stopTime = time.time()
    print("parallel execution floquet took {:.5f}".format(stopTime - startTime))

    return gf

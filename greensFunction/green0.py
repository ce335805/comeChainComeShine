import numpy as np
import sec_order.analyticalEGS as anaGS
import globalSystemParams as prms

def g0T(kPoint, tPoint):
    return - 1j * np.exp(-1j * 2. * prms.t * np.cos(kPoint[:, None]) * tPoint[None, :])


def g0VecT(kVec, tVec):
    initialState = np.zeros(prms.chainLength + 1, dtype='double')
    gs = anaGS.findGS1st(initialState, 0.)
    _, occupations = np.meshgrid(np.ones(tVec.shape), gs[0: -1])
    GF = g0T(kVec, tVec)
    GF = np.multiply(1 - occupations, GF)
    return GF



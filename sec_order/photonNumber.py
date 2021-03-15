import numpy as np
import globalSystemParams as param
import energyFunctions as eF

def avPhotonNumber1st(state, eta):
    assert (len(state) == param.chainLength + 1)
    assert (np.abs(state[-1]) < 1e-10)
    return (eta * eF.J(state[0 : -1]) / param.w0)**2

def avPhotonNumber2nd(state, eta):
    #return avPhotonNumber1st(state, eta)
    assert (len(state) == param.chainLength + 1)
    assert (np.abs(state[-1]) < 1e-10)
    gam = eF.gamma(state[0: -1], eta)
    epsilon = np.sqrt(1. - gam**2)
    u = eF.calcU(gam)
    v = eF.calcV(gam)
    vsq = np.abs(v) ** 2
    upvsq = np.abs(u + v) ** 2
    X = eta * eF.J(state[0 : -1]) / (2. * epsilon * eF.LT(state[0: -1], eta))
    return vsq + X ** 2 * upvsq

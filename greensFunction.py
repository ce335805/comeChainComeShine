import numpy as np
import sec_order.secOrder as sec
import globalSystemParams as prms
import energyFunctions as eF

def calcSpectralPoint(kPoint, wPoint, eta):
    initialState = np.zeros(prms.chainLength + 1, dtype='double')
    gs = sec.findGS(initialState, eta)
    T = eF.T(gs[0 : prms.chainLength])
    J = eF.J(gs[0 : prms.chainLength])
    wDash = prms.w0 - eta**2 * T
    gK = - 2. * eta * prms.t * np.sin(kPoint)
    eK = 2. * prms.t * np.cos(kPoint)
    eKBar = eK - gK * gK / wDash

    deltaPlus = 1e-8

    ellCutoff = 10

    spectral = 0.
    for ell in range(ellCutoff):
        lorentz = deltaPlus / ((wPoint - (eKBar - 2. * (gK / wDash) * J + ell * wDash))**2 + deltaPlus**2)
        ellFac = 1. / (np.math.factorial(ell)) * (gK/wDash)**(2 * ell)
        expPrefac = np.exp(- gK**2 / (wDash**2))
        spectral += expPrefac * ellFac * lorentz * deltaPlus
        #lorentz = deltaPlus / ((wPoint - eK)**2 + deltaPlus**2)
        #spectral += lorentz

    return np.log10(spectral)

def calcSpectral(kVec, wVec, eta):

    kVec, wVec = np.meshgrid(kVec, wVec)

    spectralFunc = calcSpectralPoint(kVec, wVec, eta)

    return spectralFunc





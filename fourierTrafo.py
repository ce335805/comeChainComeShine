import numpy as np

def FT(tVec, f):
    t0 = tVec[0]
    dt = tVec[1] - tVec[0]
    wVec = wVecFromTVec(tVec)
    w0 = wVec[0]
    dw = wVec[1] - wVec[0]
    wVecPrefac = np.arange(len(wVec))
    phaseFac = dt * np.exp(-1j * wVecPrefac * dw * t0) / np.sqrt(2. * np.pi)
    fTilde = f * np.exp(-1j * w0 * tVec)
    g = np.fft.fft(fTilde) #per default performs fft on last dimension
    return wVec, phaseFac * g


def tVecFromWVec(wVec):
    dw = wVec[1] - wVec[0]
    t0 = - np.pi / dw
    tVec = np.linspace(t0, -t0, len(wVec), endpoint=False)
    return tVec

def wVecFromTVec(tVec):
    dt = tVec[1] - tVec[0]
    w0 = - np.pi / dt
    wVec = np.linspace(w0, -w0, len(tVec), endpoint=False)
    return wVec
import numpy as np

def FT(tVec, f):
    t0 = tVec[0]
    dt = tVec[1] - tVec[0]
    w0 = - np.pi / dt
    wVec = np.linspace(w0, -w0, len(tVec), endpoint=False)
    dw = wVec[1] - wVec[0]
    #print("dw * dt = {}".format(dw * dt * len(tVec) / (2. * np.pi)))
    wVecPrefac = np.arange(len(wVec))
    phaseFac = dt * np.exp(-1j * wVecPrefac * dw * t0) / np.sqrt(2. * np.pi)
    fTilde = f * np.exp(-1j * w0 * tVec)
    g = np.fft.fft(fTilde) #per default performs fft on last dimension
    return wVec, phaseFac * g
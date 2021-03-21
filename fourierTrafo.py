import numpy as np

def FT(tVec, f):
    t0 = tVec[0]
    dt = tVec[1] - tVec[0]
    wVec = np.fft.fftfreq(tVec.size) * 2. * np.pi / dt
    phaseFac = dt * np.exp(-1j * wVec * t0) / np.sqrt(2. * np.pi)
    g = np.fft.fft(f)
    return wVec, phaseFac * g
import numpy as np

chainLength = 170
t = - 1.
w0 = 1.
numberElectrons = chainLength//2
accuracy = 1e-10
maxiter = 100
maxPhotonNumber = 100

reuseSin = np.diag(np.zeros(maxPhotonNumber))
reuseCos = np.diag(np.zeros(maxPhotonNumber))
import numpy as np
import energyFunctions
from scipy.optimize import minimize
import photonState as optPh

def findOptimalTandJ(eta):
    # print(setupPhotonHamiltonian(1.0, 1.0, eta).real)

    TBounds = energyFunctions.TBounds()
    JBounds = energyFunctions.JBounds()
    paramBounds = np.array([TBounds, JBounds])
    print(paramBounds)

    maxiter = 1e4
    optionsDict = {"maxiter": maxiter, "disp": False}
    initialGuess = np.array([0., 10.1])
    result = minimize(optPh.findSmalestEigenvalue(), initialGuess, args=eta, bounds=paramBounds, tol=prms.accuracy,
                      options=optionsDict)

    if result.success:
        print('The TJ-optimization was : --- SUCCESSFULL! ---')
    else:
        print('The TJ-optimization: --- FAILED! ---')

    return result.x
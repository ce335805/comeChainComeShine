import numpy as np
import utils
import globalSystemParams as param
import energyFunctions as eF
import arb_order.photonState as ptState
from scipy.optimize import minimize




def findGS(state, eta, orderH):
    pauliBounds = np.zeros((len(state), 2), dtype='double')
    pauliBounds[0: param.chainLength, 1] = 1.0
    maxiter = param.maxiter
    optionsDict = {"maxiter": maxiter, "disp": False}
    constraintsDict = {"type": 'eq', "fun": utils.electronNumberZero}
    result = minimize(ptState.energyFromState, state, args=(eta, orderH), bounds=pauliBounds, tol=param.accuracy, options=optionsDict, constraints=constraintsDict)

    if result.success:
        print('The optimization was : --- SUCCESSFULL! ---')
    else:
        print('The optimization: --- FAILED! ---')

    return result.x

def findPhotonNumbers(etas, orderH):

    initialState = np.zeros(param.chainLength, dtype='double')
    initialState[0: param.numberElectrons] = 1.0
    avPhotonNumbers = np.zeros((len(etas)), dtype='double')

    for indEta in range(len(etas)):
        gsTemp = findGS(initialState, etas[indEta], orderH)
        avPhotonNumbers[indEta] = ptState.avPhotonNum(gsTemp, etas[indEta], orderH)
    return avPhotonNumbers


def findGSEnergies(etas, orderH):

    initialState = np.zeros(param.chainLength, dtype='double')
    initialState[0: param.numberElectrons] = 1.0
    gsEnergies = np.zeros((len(etas)), dtype='double')

    for indEta in range(len(etas)):
        gsTemp = findGS(initialState, etas[indEta], orderH)
        gsEnergies[indEta] = ptState.energyFromState(gsTemp, etas[indEta], orderH)
    return gsEnergies



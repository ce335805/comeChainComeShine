import numpy as np
import utils
import globalSystemParams as param
import energyFunctions as eF
import arb_order.photonState as optPh
from scipy.optimize import minimize




def findGS(state, eta):
    pauliBounds = np.zeros((len(state), 2), dtype='double')
    pauliBounds[0: param.chainLength, 1] = 1.0
    maxiter = param.maxiter
    optionsDict = {"maxiter": maxiter, "disp": False}
    constraintsDict = {"type": 'eq', "fun": utils.electronNumberZero}
    result = minimize(energyFromState, state, args=eta, bounds=pauliBounds, tol=param.accuracy, options=optionsDict, constraints=constraintsDict)

    if result.success:
        print('The optimization was : --- SUCCESSFULL! ---')
    else:
        print('The optimization: --- FAILED! ---')

    return result.x

def findPhotonNumbers(etas):

    initialState = np.zeros(param.chainLength, dtype='double')
    initialState[0: param.numberElectrons] = 1.0
    avPhotonNumbers = np.zeros((len(etas)), dtype='double')

    for indEta in range(len(etas)):
        gsTemp = findGS(initialState, etas[indEta])
        avPhotonNumbers[indEta] = avPhotonNum(gsTemp, etas[indEta])
    return avPhotonNumbers


def findGSEnergies(etas):

    initialState = np.zeros(param.chainLength, dtype='double')
    initialState[0: param.numberElectrons] = 1.0
    gsEnergies = np.zeros((len(etas)), dtype='double')

    for indEta in range(len(etas)):
        gsTemp = findGS(initialState, etas[indEta])
        gsEnergies[indEta] = energyFromState(gsTemp, etas[indEta])
    return gsEnergies


def energyFromState(electronicState, eta):
    T = eF.T(electronicState)
    J = eF.J(electronicState)
    E = optPh.findSmalestEigenvalue([T, J], eta)
    return E

def photonGS(electronicState, eta):
    T = eF.T(electronicState)
    J = eF.J(electronicState)
    gs = optPh.findPhotonGS([T, J], eta)
    return gs

def avPhotonNum(electronicState, eta):
    T = eF.T(electronicState)
    J = eF.J(electronicState)
    av = optPh.averagePhotonNumber([T, J], eta)
    return av
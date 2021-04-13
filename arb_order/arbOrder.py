import numpy as np
import utils
import globalSystemParams as param
import energyFunctions as eF
import arb_order.photonState as ptState
from scipy.optimize import minimize
import initialState as ini




def findGS(eta, orderH):
    state = ini.getG0InitialStateNum()
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

    avPhotonNumbers = np.zeros((len(etas)), dtype='double')

    for indEta in range(len(etas)):
        gsTemp = findGS(etas[indEta], orderH)
        avPhotonNumbers[indEta] = ptState.avPhotonNum(gsTemp, etas[indEta], orderH)
    return avPhotonNumbers


def findGSEnergies(etas, orderH):

    gsEnergies = np.zeros((len(etas)), dtype='double')

    for indEta in range(len(etas)):
        gsTemp = findGS(etas[indEta], orderH)
        gsEnergies[indEta] = ptState.energyFromState(gsTemp, etas[indEta], orderH)
    return gsEnergies

def getPhotonGS(eta, orderH):
    gs = findGS(eta, orderH)
    ptGS = ptState.photonGS(gs, eta, orderH)
    return ptGS


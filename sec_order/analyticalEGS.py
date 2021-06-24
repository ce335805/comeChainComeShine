import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import energyFunctions as eF
import globalSystemParams as param
import utils
import sec_order.photonNumber

import initialState as ini

def findGS1st(eta):
    state = ini.getG0InitialStateAna()
    return state
    pauliBounds = np.zeros((len(state), 2), dtype='double')
    pauliBounds[0: param.chainLength, 1] = 1.0
    pauliBounds[-1, 1] = np.inf
    maxiter = param.maxiter
    optionsDict = {"maxiter": maxiter, "disp": False}
    constraintsDict = {"type": 'eq', "fun": utils.electronNumberZero}
    result = minimize(eF.hA1st, state, args=eta, bounds=pauliBounds, tol=param.accuracy, options=optionsDict, constraints=constraintsDict)

    if result.success:
        print('The optimization was : --- SUCCESSFULL! ---')
    else:
        print('The optimization: --- FAILED! ---')

    return result.x

def findGS(eta):
    state = ini.getG0InitialStateAna()
    pauliBounds = np.zeros((len(state), 2), dtype='double')
    pauliBounds[0: param.chainLength, 1] = 1.0
    pauliBounds[-1, 1] = np.inf
    maxiter = param.maxiter
    optionsDict = {"maxiter": maxiter, "disp": False}
    constraintsDict = {"type": 'eq', "fun": utils.electronNumberZero}
    result = minimize(eF.hA2nd, state, args=eta, bounds=pauliBounds, tol=param.accuracy, options=optionsDict, constraints=constraintsDict)

    if result.success:
        print('The optimization was : --- SUCCESSFULL! ---')
    else:
        print('The optimization: --- FAILED! ---')

    return result.x


def findGSExactSec(eta):
    state = ini.getG0InitialStateAna()
    return state
    pauliBounds = np.zeros((len(state), 2), dtype='double')
    pauliBounds[0: param.chainLength, 1] = 1.0
    pauliBounds[-1, 1] = np.inf
    maxiter = 1e5
    optionsDict = {"maxiter": maxiter, "disp": False}
    constraintsDict = {"type": 'eq', "fun": utils.electronNumberZero}
    result = minimize(eF.secOrderHamiltonian, state, args=eta, bounds=pauliBounds, tol=param.accuracy, options=optionsDict, constraints=constraintsDict)

    if result.success:
        print('The optimization was : --- SUCCESSFULL! ---')
    else:
        print('The optimization: --- FAILED! ---')

    return result.x


def getFSSchifts():
    etas = np.linspace(0.0, 0.2, 50)
    shifts = np.zeros((0), dtype='double')
    for currentEta in etas:
        #print('looking for GS at eta = {}'.format(currentEta))
        GS = findGS(currentEta)
        shift = findFSShift(GS)
        shifts = np.append(shifts, [shift])
        print('The boson number in the GS is ------------ {}'.format(GS[-1]))

    return shifts

def findPhotonNumberInGS2nd(etas):
    photonNumber = np.zeros((len(etas)), dtype='double')
    for indEta in np.arange(len(etas)):
        gsTemp = findGS(etas[indEta])
        print("J(GS) = {}".format(eF.J(gsTemp[0 : -1])))
        phNum = sec_order.photonNumber.avPhotonNumber2nd(gsTemp, etas[indEta])
        photonNumber[indEta] = phNum
    return photonNumber

def findPhotonNumberExactSec(etas):
    photonNumber = np.zeros((len(etas)), dtype='double')
    for indEta in np.arange(len(etas)):
        gsTemp = findGSExactSec(etas[indEta])
        phNum = sec_order.photonNumber.avPhotonNumber2nd(gsTemp, etas[indEta])
        photonNumber[indEta] = phNum
    return photonNumber

def findPhotonNumber1st(etas):
    photonNumber = np.zeros((len(etas)), dtype='double')
    for indEta in np.arange(len(etas)):
        gsTemp = findGS1st(etas[indEta])
        phNum = sec_order.photonNumber.avPhotonNumber1st(gsTemp, etas[indEta])
        photonNumber[indEta] = phNum
    return photonNumber

def findGSEnergy1st(etas):
    gsEnergy = np.zeros((len(etas)), dtype='double')
    for indEta in np.arange(len(etas)):
        gsTemp = findGS1st(etas[indEta])
        gsEnTemp = eF.firstOrderHamiltonian(gsTemp, etas[indEta])
        gsEnergy[indEta] = gsEnTemp
    return gsEnergy


def findGSEnergyExactSec(etas):
    gsEnergy = np.zeros((len(etas)), dtype='double')
    for indEta in np.arange(len(etas)):
        gsTemp = findGSExactSec(etas[indEta])
        gsEnTemp = eF.secOrderHamiltonian(gsTemp, etas[indEta])
        gsEnergy[indEta] = gsEnTemp
    return gsEnergy


def findFSShift(state):
    pos = 0
    while (True):
        if (np.abs(state[pos] - 1.) < 5. * 1e-1):
            pos = pos + 1

        else:
            break

    while (True):
        if(pos > param.chainLength - 1):
            pos = pos//2
            break
        if (np.abs(state[pos]) < 5. * 1e-1):
            pos = pos + 1
        else:
            break

    pos = pos + param.numberElectrons/2
    pos = pos//2

    fsCenter = 2 * np.pi / param.chainLength * pos
    fsCenter = fsCenter - 2. * np.pi

    return fsCenter


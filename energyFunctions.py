import numpy as np
import globalSystemParams as param
from scipy.optimize import minimize
import utils
import cmath


def h0(state):
    indices = np.arange(len(state))
    indices = 2. * param.t * np.cos(2. * np.pi / float(param.chainLength) * indices)
    return 0.5 * param.w0 + param.w0 * state[-1] + np.sum(np.multiply(state[0:-1], indices))


def TVec(state):
    kinetic = np.arange(len(state))
    kinetic = 2. * param.t * np.cos(2. * np.pi / float(param.chainLength) * kinetic)
    return kinetic


def T(state):
    return np.sum(np.multiply(TVec(state), state))


def minusT(state):
    return - T(state)


def JVec(state):
    current = np.arange(len(state))
    current = - 2. * param.t * np.sin(2. * np.pi / float(param.chainLength) * current)
    return current


def J(state):
    return np.sum(np.multiply(JVec(state), state))


def minusJ(state):
    return - J(state)


def hA2nd(state, eta):
    return param.w0 * (1. - (eta * eta / param.w0) * T(state[0 : -1])) * (state[-1] + 0.5) - \
           ((eta * eta) / param.w0) * J(state[0 : -1]) * J(state[0 : -1]) + T(state[0 : -1])


def hA2ndVec(state, eta):
    return param.w0 * (1. - (eta * eta / param.w0) * TVec(state[0 : -1])) * (state[-1] + 0.5) - \
           ((eta * eta) / param.w0) * np.multiply(JVec(state[0 : -1]), JVec(state[0 : -1])) + TVec(state[0 : -1])


def TBounds():
    initialState = np.zeros(param.chainLength, dtype='double')
    initialState[0: param.numberElectrons] = 1.0

    pauliBounds = np.zeros((len(initialState), 2), dtype='double')
    pauliBounds[:, 1] = 1.0
    maxiter = 1e3
    optionsDict = {"maxiter": maxiter, "disp": False}
    constraintsDict = {"type": 'eq', "fun": utils.electronNumberZero}
    resultMin = minimize(T, initialState, bounds=pauliBounds, tol=param.accuracy, options=optionsDict,
                         constraints=constraintsDict)

    resultMax = minimize(minusT, initialState, bounds=pauliBounds, tol=param.accuracy, options=optionsDict,
                         constraints=constraintsDict)
    if resultMin.success and resultMax.success:
        print('TBounds optimization was : --- SUCCESSFULL! ---')
    else:
        print('TBounds optimization: --- FAILED! ---')

    TMin = T(resultMin.x)
    TMax = T(resultMax.x)

    return [TMin, TMax]


def JBounds():
    initialState = np.zeros(param.chainLength, dtype='double')
    initialState[0: param.numberElectrons] = 1.0
    pauliBounds = np.zeros((len(initialState), 2), dtype='double')
    pauliBounds[:, 1] = 1.0
    maxiter = 1e3
    optionsDict = {"maxiter": maxiter, "disp": False}
    constraintsDict = {"type": 'eq', "fun": utils.electronNumberZero}
    resultMin = minimize(J, initialState, bounds=pauliBounds, tol=param.accuracy, options=optionsDict,
                         constraints=constraintsDict)

    resultMax = minimize(minusJ, initialState, bounds=pauliBounds, tol=param.accuracy, options=optionsDict,
                         constraints=constraintsDict)
    if resultMin.success and resultMax.success:
        print('JBounds optimization was : --- SUCCESSFULL! ---')
    else:
        print('JBounds optimization: --- FAILED! ---')

    JMin = J(resultMin.x)
    JMax = J(resultMax.x)

    return [JMin, JMax]


def firstOrderHamiltonian(state, eta):
    eState = state[0 : param.chainLength]

    H = param.w0 * (state[-1] + .5) \
        - eta**2 / param.w0 * J(eState)**2 \
        + T(eState)

    return H.real

def secOrderHamiltonian(state, eta):
    #return firstOrderHamiltonian(state, eta)
    eState = state[0 : param.chainLength]
    gam = gamma(eState, eta)
    epsilon = cmath.sqrt(1. - gam**2)
    upvsqr = np.abs((calcU(gam) + calcV(gam)))**2

    H = 2. * epsilon * LT(eState, eta) * (state[-1] + .5) \
        - (eta * J(eState))**2 * upvsqr / (2. * epsilon * LT(eState, eta)) \
        + T(eState)

    assert(np.abs(np.imag(H)) < 1e-14)

    return H.real
    
    
def LT(electronicState, eta):
    return .5 * param.w0 - .5 * eta**2 * T(electronicState)

def calcU(gam):
    epsilon = cmath.sqrt((1. + 0j) - gam**2)
    u = (cmath.sqrt(1 + epsilon)) / (cmath.sqrt(2. * epsilon))
    return u

def calcV(gam):
    epsilon = cmath.sqrt((1. + 0j) - gam**2)
    v = - gam / (cmath.sqrt(2. * epsilon) * cmath.sqrt(1 + epsilon))
    return v


def gamma(electronicState, eta):
    gamma = eta**2 * T(electronicState) / (eta**2 * T(electronicState) - param.w0)
    return gamma
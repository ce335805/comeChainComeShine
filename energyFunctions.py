import numpy as np
import globalSystemParams as param
from scipy.optimize import minimize
import utils
import cmath


def h0(state):
    indices = np.arange(len(state))
    indices = 2. * param.t * np.cos(2. * np.pi / float(param.chainLength) * indices)
    return 0.5 * param.w0 + param.w0 * state[-1] + np.sum(np.multiply(state[0:-1], indices))

#input only electronic part of state
def TVec(state):
    kinetic = np.arange(len(state))
    kinetic = 2. * param.t * np.cos(2. * np.pi / float(param.chainLength) * kinetic)
    return kinetic


#input only electronic part of state
def T(state):
    return np.sum(np.multiply(TVec(state), state))


#input only electronic part of state
def minusT(state):
    return - T(state)


def JVec(electronicState):
    current = np.arange(len(electronicState))
    current = - 2. * param.t * np.sin(2. * np.pi / float(param.chainLength) * current)
    return current


def J(electronicState):
    return np.sum(np.multiply(JVec(electronicState), electronicState))


def minusJ(electronicState):
    return - J(electronicState)

def hA1st(state, eta):
    return param.w0 * (state[-1] + 0.5) + T(state[0 : -1]) - \
           ((eta * eta) / param.w0) * J(state[0 : -1]) * J(state[0 : -1])

def hA2nd(state, eta):
    return param.w0 * (1. - (eta * eta / param.w0) * T(state[0 : -1])) * (state[-1] + 0.5) - \
           ((eta * eta) / param.w0) * J(state[0 : -1]) * J(state[0 : -1]) + T(state[0 : -1])


def hA2ndVec(state, eta):
    return param.w0 * (1. - (eta * eta / param.w0) * TVec(state[0 : -1])) * (state[-1] + 0.5) - \
           ((eta * eta) / param.w0) * np.multiply(JVec(state[0 : -1]), JVec(state[0 : -1])) + TVec(state[0 : -1])



def firstOrderHamiltonian(state, eta):
    eState = state[0 : param.chainLength]

    H = param.w0 * (state[-1] + .5) \
        - eta**2 / param.w0 * J(eState)**2 \
        + T(eState)

    return H

def secOrderHamiltonian(state, eta):
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
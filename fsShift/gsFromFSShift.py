import numpy as np
import globalSystemParams as prms
from arb_order import photonState
import matplotlib.pyplot as plt
from arb_order import photonState
from sec_order import photonNumber
from scipy.optimize import minimize
from scipy.optimize import minimize_scalar
import matplotlib.pyplot as plt


def occupationsForLengthsZeroShift(Ls, etasNonNorm, orderH):
    originalL = prms.chainLength
    occupationsLs = np.zeros((len(Ls), len(etasNonNorm)), dtype='double')
    for indL, l in enumerate(Ls):
        prms.chainLength = l
        etas = etasNonNorm / np.sqrt(l)
        occs = occupationForLZeroShift(etas, orderH)
        occupationsLs[indL, :] = occs
    prms.chainLength = originalL
    return occupationsLs

def occupationForLZeroShift(etas, orderH):
    ptNs = np.zeros(len(etas), dtype = 'double')

    for indEta, eta in enumerate(etas):
        T = prms.t / (np.pi) * (np.sin(np.pi / 2.) - np.sin(-np.pi / 2.)) * prms.chainLength
        J = 0.
        if(orderH == 1):
            nPt = eta**2 / prms.w0**2 * J**2
        elif (orderH == 2):
            nPt = photonNumber.avPhotonNumber2ndTJ(T, J, eta)
        else:
            nPt = photonState.averagePhotonNumber([T, J], eta, 3)
        ptNs[indEta] = nPt
    return ptNs


def occupationsForLengths(Ls, etasNonNorm, orderH, bins):
    originalL = prms.chainLength
    occupationsLs = np.zeros((len(Ls), len(etasNonNorm)), dtype='double')
    for indL, l in enumerate(Ls):
        prms.chainLength = l
        etas = etasNonNorm / np.sqrt(l)
        occs = occupationsFromShifts(etas, orderH, bins)
        occupationsLs[indL, :] = occs
    prms.chainLength = originalL
    return occupationsLs

def occupationsFromShifts(etas, orderH, bins):
    minima = energyMinimaNum(etas, orderH, bins)
    ptNs = np.zeros(len(etas), dtype = 'double')

    print("minima = {}".format(minima))

    for indEta, eta in enumerate(etas):
        T = prms.t / (np.pi) * (np.sin(np.pi / 2. + minima[indEta]) - np.sin(-np.pi / 2. + minima[indEta])) * prms.chainLength
        J = prms.t / (np.pi) * (np.cos(np.pi / 2. + minima[indEta]) - np.cos(-np.pi / 2. + minima[indEta])) * prms.chainLength
        nPt = -1.
        if(orderH == 1):
            nPt = eta**2 / prms.w0 * J**2
        elif (orderH == 2):
            nPt = photonNumber.avPhotonNumber2ndTJ(T, J, eta)
        ptNs[indEta] = nPt
    return ptNs

def energyMinimaNum(etas, orderH, bins):

    landscapes = getManyEnergyLandscapes(etas, orderH, bins)
    minimaKs = np.zeros(len(etas), dtype = 'double')
    for indEta, eta in enumerate(etas):
        minimumPos = np.where(np.abs(landscapes[indEta, :] - np.amin(landscapes[indEta, :])) < 1e-8)
        minimumK = np.linspace(0., .5 * np.pi, bins)[minimumPos[0][0]]
        minimaKs[indEta] = minimumK
        #fig = plt.figure()
        #ax = fig.add_subplot(111)
        #ax.plot(np.linspace(0., .5 * np.pi, bins), landscapes[indEta, :])
        #plt.show()

    return minimaKs

def energyMinima(etas, orderH, bins):

    minimaKs = np.zeros(len(etas), dtype = 'double')
    for indEta, eta in enumerate(etas):
        minimumK = findMinimalShift(eta, orderH)
        minimaKs[indEta] = minimumK
    return minimaKs

def getManyEnergyLandscapes(etas, orderH, bins):
    landscapes = np.zeros((len(etas), bins))
    for indEta in range(len(etas)):
        landscapes[indEta, :] = getEnergyLandscape(etas[indEta], orderH, bins)

    return landscapes

def getEnergyLandscape(eta, orderH, bins):
    xArr = np.linspace(0., 2. * np.pi, bins)
    eArr = energyLandscapeShifts(eta, xArr, orderH)

    return eArr


def energyLandscapeShifts(eta, xArr, orderH):
    eArr = np.zeros((len(xArr)), dtype='double')
    for indX in range(len(xArr)):
        eArr[indX] = eFromShift(xArr[indX], eta, orderH)

    return eArr

def eFromShift(x, eta, orderH):
    T = prms.t / (np.pi) * (np.sin(np.pi / 2. + x) - np.sin(-np.pi / 2. + x)) * prms.chainLength
    J = prms.t / (np.pi) * (np.cos(np.pi / 2. + x) - np.cos(-np.pi / 2. + x)) * prms.chainLength
    E = -1
    if(orderH == 1):
        intTerm = eta ** 2 / prms.w0 * J ** 2
        return .5 * prms.w0 + T - intTerm
    elif(orderH == 2):
        corr = 2. * eta * eta / prms.w0 * T
        if(corr > 1):
            return - 1e10
        fac = np.sqrt(1 - corr)
        gsEAna = .5 * fac * prms.w0 + T - np.sqrt(fac) * eta * eta / (fac * prms.w0) * J * J
        #gsENum = photonState.findSmalestEigenvalue([T, J], eta, orderH)
        #print("gsEAna - gsENum = {}".format(gsEAna - gsENum))
        return gsEAna
    else:
        E = photonState.findSmalestEigenvalue([T, J], eta, orderH)
    return E


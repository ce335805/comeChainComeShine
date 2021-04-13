import numpy as np
import globalSystemParams as prms
from arb_order import photonState
import matplotlib.pyplot as plt

def plotLandscapes(etas, orderH):
    bins = 150
    landscapes = getManyEnergyLandscapes(etas, orderH, bins)

    fig, ax = plt.subplots(nrows=1, ncols=1)
    cmap = plt.cm.get_cmap('terrain')
    xArr = np.linspace(0., 2. * np.pi, bins)
    for indEta in range(len(etas)):
        eta = etas[indEta]
        color = cmap(eta / (etas[-1] + 0.1))
        ax.plot(xArr, landscapes[indEta, :], color = color, label = r'g = {:.2f}'.format(eta))

    labelString = "$\omega$ = {:.2f}".format(prms.w0)
    ax.text(0., .5, labelString, fontsize = 14)
    plt.legend()
    plt.show()

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
    T = tFromShift(x)
    J = jFromShift(x)
    E = photonState.findSmalestEigenvalue([T, J], eta, orderH)
    return E

def tFromShift(x):
    N = prms.chainLength
    kVec = np.linspace(-np.pi / 2. + x, np.pi / 2. + x, N, endpoint=False)
    eK = 2. * prms.t * np.cos(kVec)
    return np.sum(eK)

def jFromShift(x):
    N = prms.chainLength
    kVec = np.linspace(-np.pi / 2. + x, np.pi / 2. + x, N, endpoint=False)
    eK = - 2. * prms.t * np.sin(kVec)
    return np.sum(eK)

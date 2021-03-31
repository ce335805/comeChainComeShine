import numpy as np
import globalSystemParams as prms
from arb_order import photonState
import matplotlib.pyplot as plt

def plotLandscapes(etas):
    landscapes = getManyEnergyLandscapes(etas)

    fig, ax = plt.subplots(nrows=1, ncols=1)
    cmap = plt.cm.get_cmap('terrain')
    xArr = np.linspace(0., 2. * np.pi, 50)
    for indEta in range(len(etas)):
        eta = etas[indEta]
        color = cmap(eta / (etas[-1] + 0.1))
        ax.plot(xArr, landscapes[indEta, :], color = color, label = r'g = {:.2f}'.format(eta))

    labelString = "$\omega$ = {:.2f}".format(prms.w0)
    ax.text(0., .5, labelString, fontsize = 14)
    plt.legend()
    plt.show()

def getManyEnergyLandscapes(etas):
    bins = 50
    landscapes = np.zeros((len(etas), bins))
    for indEta in range(len(etas)):
        landscapes[indEta, :] = getEnergyLandscape(etas[indEta], bins)

    return landscapes

def getEnergyLandscape(eta, bins):
    xArr = np.linspace(0., 2. * np.pi, bins)
    eArr = energyLandscapeShifts(eta, xArr)

    return eArr


def energyLandscapeShifts(eta, xArr):
    eArr = np.zeros((len(xArr)), dtype='double')
    for indX in range(len(xArr)):
        eArr[indX] = eFromShift(xArr[indX], eta)

    return eArr


def eFromShift(x, eta):
    T = tFromShift(x)
    J = jFromShift(x)
    E = photonState.findSmalestEigenvalue([T, J], eta, 3)
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

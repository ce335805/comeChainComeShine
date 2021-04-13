import numpy as np
import globalSystemParams as prms
from arb_order import arbOrder

import matplotlib.pyplot as plt

def plotPhotonOcc(lArr, etaArr, orderH):

    photonOcc = getPhotonOccupancies(lArr, etaArr, orderH)

    fig, ax = plt.subplots(nrows=1, ncols=1)
    cmap = plt.cm.get_cmap('terrain')
    for indL in range(len(lArr)):
        L = lArr[indL]
        print("L = {}".format(L))
        color = cmap(L / (lArr[-1] + 100))
        ax.plot(etaArr, photonOcc[indL, :], color=color, label=r'L = {}'.format(L))

    labelString = "$\omega$ = {:.1f}".format(prms.w0)
    ax.text(0., .5, labelString, fontsize=14)
    plt.legend()
    plt.show()



def getPhotonOccupancies(lArr, etaArr, orderH):
    photonOcc = np.zeros((len(lArr), len(etaArr)))

    #save previous model
    prevL = prms.chainLength
    prevOcc = prms.numberElectrons

    for indL in range(len(lArr)):
        #reset model
        print("Looking at L = {}".format(lArr[indL]))
        prms.chainLength = lArr[indL]
        prms.numberElectrons = prms.chainLength // 2
        etaArrRescaled = etaArr / np.sqrt(lArr[indL])

        occ = arbOrder.findPhotonNumbers(etaArrRescaled, orderH)
        photonOcc[indL, :] = occ

    #reset to previous model
    prms.chainLength = prevL
    prms.numberElectrons = prevOcc

    return photonOcc
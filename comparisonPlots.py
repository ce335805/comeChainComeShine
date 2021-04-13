import numpy as np
import matplotlib.pyplot as plt
import globalSystemParams as prms

#def compareEStates(state1, state2):


def compareArrays(x, y1, y2):
    fig, ax = plt.subplots(nrows=1, ncols=1)
#    ax.plot(x, y1, marker ='x', color = 'b', linestyle = '-', label = r"GF - 1st-order")
#    ax.plot(x, y2, marker = 'x', color = 'r', linestyle = '--', label = r"Gf - $\infty$-order")
    ax.plot(x, y1, marker = '', color = 'skyblue', linestyle = '-')
#    plt.legend()
    plt.show()

def compareArraysLog(x, y1, y2):
    y1 = np.abs(y1) + 1e-16
    y2 = np.abs(y2) + 1e-16
    fig, ax = plt.subplots(nrows=1, ncols=1)
    ax.plot(x, y1, marker = '', color = 'skyblue', linestyle = '-', label = '$A(\omega)$')
    ax.plot(x, y2, marker = '', color = 'wheat', linestyle = '-', label = '$A(\omega)^{1st}$')
    ax.set_yscale('log')
    ax.set_yticks([1e0, 2. * 1e-1, 1e-1])
    ax.set_yticklabels(['$10^0$', '$2 x 10^{-1}$','$10^{-1}$'])
#    ax.hlines(0.1735, -10., 10., colors=['gray'], label = 'L = 30')
    labelString = "L = {:.0f} \n$\omega$ = {:.1f}".format(prms.chainLength, prms.w0)
    ax.text(-7.5, 2., labelString, fontsize = 14)
    plt.legend()
    plt.show()


def plotTwoEGS(state1, state2):
    kVec = np.linspace(0, 2. * np.pi, prms.chainLength, endpoint=False)
    energyVec1 = np.cos(kVec)
    energyVec2 = np.cos(kVec) + 0.2

    fig, ax = plt.subplots(nrows=1, ncols=1)
    cmapR = plt.get_cmap('Reds')
    cmapB = plt.get_cmap('Blues')
    colors1 = cmapR(0.95 * state1[::2] + 0.05)
    colors2 = cmapB(0.95 * state2[::2] + 0.05)
    ax.scatter(kVec, energyVec1, color=colors1)
    ax.scatter(kVec, energyVec2, color=colors2)
    plt.show()
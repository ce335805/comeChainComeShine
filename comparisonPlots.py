import numpy as np
import matplotlib.pyplot as plt
import globalSystemParams as param

#def compareEStates(state1, state2):


def compareArrays(x, y1, y2):
    fig, ax = plt.subplots(nrows=1, ncols=1)
    ax.plot(x, y1, marker ='x', color = 'b', linestyle = '-', label = r"GF - 1st-order")
    ax.plot(x, y2, marker = 'x', color = 'r', linestyle = '--', label = r"Gf - $\infty$-order")
    plt.legend()
    plt.show()


def plotTwoEGS(state1, state2):
    kVec = np.linspace(0, 2. * np.pi, param.chainLength, endpoint=False)
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
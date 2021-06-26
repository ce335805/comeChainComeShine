import numpy as np
import matplotlib.pyplot as plt
import globalSystemParams as prms

#def compareEStates(state1, state2):


def compareArrays(x, y1, y2):
    fig, ax = plt.subplots(nrows=1, ncols=1)
    ax.plot(x, y1, marker = '', color = 'skyblue', linestyle = '-')
    ax.plot(x, y2, marker='', color='black', linestyle='-', linewidth = .5)
    ax.set_xlim(-3., 3.)
    #ax.plot(x, y3, marker='', color='wheat', linestyle='--')
#    plt.legend()
    plt.show()

def compareArraysLog(x, y1, y2, y3, y4):
    y1 = np.abs(y1) + 1e-16
    y2 = np.abs(y2) + 1e-16
    y3 = np.abs(y3) + 1e-16
    y4 = np.abs(y4) + 1e-16
    fig, ax = plt.subplots(nrows=1, ncols=1)
    ax.plot(-x, y1, marker = '', color = 'lightsteelblue', linestyle = '-', label = '$1/L$ - $\pi / 2$')
    ax.plot(-x, y2, marker = '', color = 'tan', linestyle = '--', label = '$1/L$ - $\pi$')
    ax.plot(-x, y3, marker = '', color = 'red', linestyle = '--', label = '$1/L^2$ - $\pi / 2$')
    ax.plot(-x, y4, marker = '', color = 'limegreen', linestyle = '--', label = '$1/L^2$ - $\pi$')
    ax.set_yscale('log')
    #ax.set_yticks([1e0, 1e-1, 1e-2, 1e-3])
    #ax.set_yticklabels(['$10^0$','$10^{-1}$','$10^{-2}$','$10^{-3}$'])
#    ax.hlines(0.1735, -10., 10., colors=['gray'], label = 'L = 30')
    labelString = "L = {:.0f} \n$\omega$ = {:.1f}".format(prms.chainLength, prms.w0)
    ax.text(-7.5, 2., labelString, fontsize = 14)
    plt.legend()
    plt.show()
    #saveString = "AwTavEQ" + str(tAv) + ".pdf"
    #plt.savefig(saveString)

def finiteSizeErrors(x, e1, e2, e3, e4):
    fig, ax = plt.subplots(nrows=1, ncols=1)
    ax.loglog(x, e1, marker = 'x', color = 'lightsteelblue', linestyle = '-', label = 'Mean Error - 1', linewidth = 1.)
    ax.loglog(x, e2, marker = 'x', color = 'tan', linestyle = '-', label = 'Max Error - 1', linewidth = 1.)
    ax.loglog(x, e3, marker = 'x', color = 'red', linestyle = '-', label = 'Mean error', linewidth = 1., markersize = 7.)
    ax.loglog(x, e4, marker = 'x', color = 'mediumseagreen', linestyle = '-', label = 'Max error', linewidth = 1., markersize = 7.)
    ax.loglog(x, 1. / (x * np.sqrt(x)) * 2. * 1e1 , marker = '', color = 'black', linestyle = '--', label = r'$\sim 1 / L^{\frac{3}{2}}$')
    ax.loglog(x, 1. / (x) , marker = '', color = 'gray', linestyle = '--', label = r'$\sim 1 / L$')
    ax.loglog(x, 1. / (x**2) * 1e3, marker = '', color = 'grey', linestyle = '--', label = r'$\sim 1 / L^2$')
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
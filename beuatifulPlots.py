import numpy as np
import matplotlib.pyplot as plt
import globalSystemParams as prms
from matplotlib import ticker
from matplotlib.colors import LogNorm
from coherentState import coherentState
from floquet import spectralFunction
from nonEqGreen import nonEqGreen

def plotSpec(kVec, wVec, spec):

    spec = np.roll(spec, prms.chainLength // 2 - 1, axis=1)
    kVecPosNeg = np.linspace(np.pi, -np.pi, prms.chainLength, endpoint=False)
    kVecPosNeg = np.flip(kVecPosNeg)

    fig = plt.figure()
    ax = fig.add_subplot(111)
    CS = ax.contourf(kVecPosNeg, -wVec, spec, 500, cmap = 'gnuplot2_r')
    fig.colorbar(CS, ax=ax)
    plt.show()

def plotSpecLog(kVec, wVec, spec):

    spec = spec + 1e-5

    spec = np.roll(spec, prms.chainLength // 2 - 1, axis=1)
    spec = np.abs(spec)
    kVecPosNeg = np.linspace(np.pi, -np.pi, prms.chainLength, endpoint=False)
    kVecPosNeg = np.flip(kVecPosNeg)

    fig = plt.figure()
    ax = fig.add_subplot(111)
    lvls = np.logspace(np.log10(spec[-1, -1]) - 0.01, 0, 200)
    CS = ax.contourf(kVecPosNeg, -wVec, spec, cmap = 'pink', norm=LogNorm(), levels = lvls)
    cbar = fig.colorbar(CS, ax=ax)
    cbar.set_ticks([1e-0, 1e-1, 1e-2])
    cbar.set_ticklabels([r'$10^{0}$', r'$10^{-1}$', r'$10^{-2}$'])
    #ax.arrow(0., 2. * prms.t, 0., - 4. * prms.t, length_includes_head = True, color = 'white', width = 0.025, head_width = 0.11, head_length = 0.015)
    #ax.arrow(np.pi/2, 0., 0., prms.w0, length_includes_head = True, color = 'white', width = 0.025, head_width = 0.11, head_length = 0.015)
    plt.show()

    print('{:.3f}'.format(spec[0, 0]))

def plotPtGS(ptGS, eta):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    bins = np.arange(len(ptGS))
    ax.bar(bins, np.abs(ptGS), log = True, color = 'wheat')
    ax.hlines(1., -2., len(ptGS) + 1, linestyles = '--', colors='gray')
    ax.set_xlim(-1, 31)
    ax.set_ylim(1e-12, 1e1)
    labelString = "g = {:.2f} \n $\omega$ = {:.2f}".format(eta, prms.w0)
    ax.text(20, 1e-4, labelString, fontsize = 20)
    plt.show()

def plotPtGSWithCoh(ptGS, N, eta):
    cohState = coherentState.getCoherentStateForN(N)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    bins = np.arange(len(ptGS))
    ax.bar(bins, np.abs(ptGS), log = True, color = 'wheat')
    ax.plot(np.abs(cohState), color = 'red')
    ax.hlines(1., -2., len(ptGS) + 1, linestyles = '--', colors='gray')
    ax.set_xlim(-1, 51)
    ax.set_ylim(1e-20, 1e1)
    labelString = "g = {:.2f} \n $\omega$ = {:.2f}".format(eta, prms.w0)
    ax.text(20, 1e-4, labelString, fontsize = 20)
    plt.show()


def calculateAndPlotShakeOffs():

    tau = 2. * np.pi / prms.w0
    wVec = np.linspace(-10., 10., 1000, endpoint=False)
    tAv = np.linspace(0. * tau, 19. * tau, 200, endpoint=False)
    kVec = np.array([np.pi / 2])
    print("kVec-Val = {}".format(kVec[0] / np.pi))
    damping = .1

    fig, ax = plt.subplots(nrows=1, ncols=1)
    cmap = plt.cm.get_cmap('terrain')

    LArr = np.array([12, 22])
    for lInd, lVal in enumerate(LArr):

        color = cmap(lVal / (LArr[-1] + 30))

        prms.chainLength = lVal
        prms.numberElectrons = lVal // 2
        cohN = lVal / 100.
        eta = .1 / np.sqrt(lVal)
        prms.maxPhotonNumber = int(10 + cohN)

        gfNonEqCoh = nonEqGreen.gfCohWLesser(kVec, wVec, tAv, eta, damping, cohN)
        gfNonEqCohN0 = 1. / (20. * tau) * (tAv[1] - tAv[0]) * np.sum(gfNonEqCoh, axis=2)
        labelStr = "L = {:.0f}".format(lVal)
        ax.plot(wVec, np.abs(np.imag(gfNonEqCohN0[0, :])), color=color, linestyle='-', label = labelStr)

        if (lInd == len(LArr) - 1):
            gWFloquet = spectralFunction.gLesserW(kVec, wVec, tAv, eta, cohN, damping)
            gWFloquetInt = 1. / (20. * tau) * (tAv[1] - tAv[0]) * np.sum(gWFloquet, axis=2)
            ax.plot(wVec, np.abs(np.imag(gWFloquetInt[0, :])), color = 'gray', linestyle = '--', label = 'Floquet')

    ax.set_yscale('log')
    ax.set_yticks([1e0, 1e-1, 1e-2])
    ax.set_yticklabels(['$10^0$', '$10^{-1}$', '$10^{-2}$'])
    plt.legend()
    plt.show()

import numpy as np
import matplotlib.pyplot as plt
import globalSystemParams as prms
from matplotlib import ticker
from matplotlib.colors import LogNorm
from coherentState import coherentState

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
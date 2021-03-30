import numpy as np
import matplotlib.pyplot as plt
import globalSystemParams as prms
from matplotlib import ticker
from matplotlib.colors import LogNorm

def plotSpec(kVec, wVec, spec):

    spec = np.roll(spec, prms.chainLength // 2, axis=1)
    kVecPosNeg = np.linspace(np.pi, -np.pi, prms.chainLength, endpoint=False)
    kVecPosNeg = np.flip(kVecPosNeg)

    fig = plt.figure()
    ax = fig.add_subplot(111)
    CS = ax.contourf(kVecPosNeg, -wVec, spec, 500, cmap = 'gnuplot2')
    fig.colorbar(CS, ax=ax)
    plt.show()

def plotSpecLog(kVec, wVec, spec):

    spec = np.roll(spec, prms.chainLength // 2, axis=1)
    spec = np.abs(spec) + 1e-10
    kVecPosNeg = np.linspace(np.pi, -np.pi, prms.chainLength, endpoint=False)
    kVecPosNeg = np.flip(kVecPosNeg)

    fig = plt.figure()
    ax = fig.add_subplot(111)
    lvls = np.logspace(-12, 0, 200)
    CS = ax.contourf(kVecPosNeg, -wVec, spec, cmap = 'gnuplot2', norm=LogNorm(), levels = lvls)
    fig.colorbar(CS, ax=ax)
    plt.show()

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

def plotPtGSWithCoh(ptGS, cohState, eta):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    bins = np.arange(len(ptGS))
    ax.bar(bins, np.abs(ptGS), log = True, color = 'wheat')
    ax.plot(np.abs(cohState), color = 'red')
    ax.hlines(1., -2., len(ptGS) + 1, linestyles = '--', colors='gray')
    ax.set_xlim(-1, 31)
    ax.set_ylim(1e-12, 1e1)
    labelString = "g = {:.2f} \n $\omega$ = {:.2f}".format(eta, prms.w0)
    ax.text(20, 1e-4, labelString, fontsize = 20)
    plt.show()
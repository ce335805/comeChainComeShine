import numpy as np
import matplotlib.pyplot as plt
import globalSystemParams as prms
from matplotlib import ticker
from matplotlib.colors import LogNorm
from coherentState import coherentState
from floquet import spectralFunction
from nonEqGreen import nonEqGreen
import matplotlib.patches as patches
import matplotlib as mpl
import fsShift.gsFromFSShift as fsShift

import h5py

mpl.rcParams['font.family'] = 'sans-serif'
mpl.rcParams['lines.linewidth'] = 2
mpl.rcParams['lines.markersize'] = 10
mpl.rcParams['font.size'] = 10  # <-- change fonsize globally
mpl.rcParams['legend.fontsize'] = 10
mpl.rcParams['axes.titlesize'] = 10
mpl.rcParams['axes.labelsize'] = 10
mpl.rcParams['xtick.major.size'] = 3
mpl.rcParams['ytick.major.size'] = 3
mpl.rcParams['xtick.major.width'] = .7
mpl.rcParams['ytick.major.width'] = .7
mpl.rcParams['xtick.direction'] = 'out'
mpl.rcParams['ytick.direction'] = 'out'
mpl.rcParams['figure.titlesize'] = 10
#mpl.rcParams['figure.figsize'] = [7.,5]
mpl.rcParams['text.usetex'] = True

fontsize = 10

def plotSpec(kVec, wVec, spec):
    spec = np.roll(spec, prms.chainLength // 2 - 1, axis=1)
    kVecPosNeg = np.linspace(np.pi, -np.pi, prms.chainLength, endpoint=False)
    kVecPosNeg = np.flip(kVecPosNeg)

    fig = plt.figure()
    ax = fig.add_subplot(111)
    CS = ax.contourf(kVecPosNeg, -wVec, spec, 500, cmap='gnuplot2_r')
    fig.colorbar(CS, ax=ax)
    plt.show()


def plotSpecLog(wVec, spec, eta):
    spec = spec + 1e-12

    spec = np.roll(spec, prms.chainLength // 2 - 1, axis=1)
    spec = np.abs(spec)
    kVecPosNeg = np.linspace(np.pi, -np.pi, prms.chainLength, endpoint=False)
    kVecPosNeg = np.flip(kVecPosNeg)

    fig = plt.figure()
    fig.set_size_inches(0.5 * 16. / 2., 0.5 * 9. / 2.)

    ax = fig.add_subplot(111)

#turn shake-off
#    left, bottom, width, height = [0.1275, 0.115, 0.615, 0.25]
    left, bottom, width, height = [0.1275, 0.625, 0.615, 0.25]
    axIn1 = fig.add_axes([left, bottom, width, height])
    #axIn1.set_yscale('log')

#turn shake-off
#    left, bottom, width, height = [0.25, 0.205, 0.15, 0.15]
    left, bottom, width, height = [0.2, 0.715, 0.15, 0.15]
    axIn2 = fig.add_axes([left, bottom, width, height])

#turn shake-off
#    left, bottom, width, height = [0.425, 0.205, 0.15, 0.15]
    left, bottom, width, height = [0.45, 0.715, 0.15, 0.15]
    axIn3 = fig.add_axes([left, bottom, width, height])

    lvls = np.logspace(np.log10(spec[-1, -1]) - 0.01, 0, 200)
    #CS = ax.contourf(kVecPosNeg, -wVec, spec, cmap='pink', norm=LogNorm(), levels=lvls)
    CS = ax.pcolormesh(kVecPosNeg, -wVec, spec, cmap='pink', norm=LogNorm(), shading = 'gouraud')
    #ax.plot(kVecPosNeg, 2. * prms.t * np.cos(kVecPosNeg), color = 'red')
    cbar = fig.colorbar(CS, ax=ax)
    cbar.set_ticks([])
    cbar.ax.minorticks_off()
    #cbar.set_ticks([1e-0, 1e-1, 1e-2])
    #cbar.set_ticklabels([r'$10^{0}$', r'$10^{-1}$', r'$10^{-2}$'])

    kPoint = np.pi / 2.
    gsT = - 2. / np.pi * prms.chainLength
    z = 2. * eta ** 2 / prms.w0 * gsT
    wTilde = prms.w0 * np.sqrt(1. - z)
    wDash = prms.w0 * (1. - z)
    fac = .5 * (2. - z) / (1. - z)
    #selfE = - fac * eta ** 2 / wDash * (- 2. * prms.t * np.sin(kPoint) ** 2)
    selfE = - eta**2 / (wTilde) * np.sqrt(np.sqrt(1 - z)) * (- 2. * prms.t * np.sin(kPoint))**2


    dynLoc = 1 - eta**2 / 2. * wTilde / prms.w0

    ax.arrow(-np.pi/2, selfE, 0., -prms.w0, length_includes_head = True, color = 'darkseagreen', width = 0.025, head_width = 0.11, head_length = 0.15)

    #turn shake-off
    #ax.set_ylim(-6, 4.5)
    ax.set_ylim(-4.1, 7.5)

    ax.set_xticks([-np.pi / 2., 0., np.pi / 2., np.pi])
    ax.set_xticklabels([r'$-\frac{\pi}{2}$', r'0', r'$\frac{\pi}{2}$', r'$\pi$'])
    ax.tick_params(axis='both', which='major', labelsize=fontsize)

    ax.set_xlabel('$k$', fontsize=fontsize)
    ax.set_ylabel('$\omega$', fontsize=fontsize, labelpad = -1)


    axIn1.plot(-wVec, spec[:, -1], color = 'lightsteelblue', linewidth = 1.5)
    axIn1.plot(-wVec, spec[:, prms.chainLength//4], color = 'tan', linewidth = 1.5)

    #turn shake-off
    #xLimBot = -.4
    xLimBot = -2.
    xLimTop = 2.5

    yLimBot = spec[-1, prms.chainLength //2] - 0.0001
    yLimTop = spec[-1, prms.chainLength //2] + .4 # + 0.01 for second shakeoff
    axIn1.set_xlim(xLimBot, xLimTop)
    axIn1.set_ylim(yLimBot, yLimTop)
    axIn1.set_yticks([])
    axIn1.set_xticks([])
    axIn1.vlines(2. * np.cos(np.pi / 2. + np.pi / prms.chainLength), yLimBot, yLimTop, color = 'black', linestyle = '--', linewidth = 1.)
    axIn1.vlines(2. * np.cos(np.pi / 2. + np.pi / prms.chainLength) + selfE, yLimBot, yLimTop, color = 'tan', linestyle = '--', linewidth = 1.)
    axIn1.vlines(2., yLimBot, yLimTop, color = 'black', linestyle = '--', linewidth = 1.)
    axIn1.vlines(2. * dynLoc, yLimBot, yLimTop, color = 'lightsteelblue', linestyle = '--', linewidth = 1.)

#    ax.vlines(np.pi - 0.05, xLimBot, xLimTop, color = 'lightsteelblue', linestyles='dotted', linewidth = 4)

    rect = patches.Rectangle((np.pi - .11, xLimBot), .1, xLimTop - xLimBot, linewidth=1., edgecolor='lightsteelblue', facecolor='none', zorder=100)
    ax.add_patch(rect)

#    ax.vlines(np.pi / 2., xLimBot, xLimTop, color = 'tan', linestyles='dotted', linewidth = 4)

    rect = patches.Rectangle((np.pi/2. - .05, xLimBot), .1, xLimTop - xLimBot, linewidth=1.1, edgecolor='tan', facecolor='none', zorder=100)
    ax.add_patch(rect)

    axIn2.plot(-wVec, spec[:, prms.chainLength//4], color = 'tan', linewidth = 1.0)
    xLimBot = -.08
    xLimTop = .02
    yLimBot = spec[-1, prms.chainLength //2] - 0.0001
    yLimTop = spec[-1, prms.chainLength //2] + 7. # + 0.01 for second shakeoff
    axIn2.set_xlim(xLimBot, xLimTop)
    axIn2.set_ylim(yLimBot, yLimTop)
    axIn2.vlines(2. * np.cos(np.pi / 2. + np.pi / prms.chainLength), yLimBot, yLimTop, color = 'black', linestyle = 'dotted', linewidth = 1.)
    axIn2.vlines(2. * np.cos(np.pi / 2. + np.pi / prms.chainLength) + selfE, yLimBot, yLimTop, color = 'tan', linestyle = 'dotted', linewidth = 1.)

    axIn2.set_xticks([-0.05, 0.])
    axIn2.set_xticklabels([r'$-0.05$', r'$0.0$'], fontsize = fontsize - 3)
    axIn2.set_yticks([])

    axIn3.plot(-wVec, spec[:, -1], color = 'lightsteelblue', linewidth = 1.)
    xLimBot = 1.95
    xLimTop = 2.05
    yLimBot = spec[-1, prms.chainLength //2] - 0.0001
    yLimTop = spec[-1, prms.chainLength //2] + 7. # + 0.01 for second shakeoff
    axIn3.set_xlim(xLimBot, xLimTop)
    axIn3.set_ylim(yLimBot, yLimTop)
    axIn3.vlines(2., yLimBot, yLimTop, color = 'black', linestyle = 'dotted', linewidth = 1.)
    axIn3.vlines(2. * dynLoc, yLimBot, yLimTop, color = 'lightsteelblue', linestyle = 'dotted', linewidth = 1.)

    axIn3.set_yticks([])
    axIn3.set_xticks([1.97, 2., 2.03])
    axIn3.set_xticklabels([r'$1.97$', r'$2$', r'$2.03$'], fontsize = fontsize - 3)

    cbar.ax.set_ylabel(r'$\log(A(k, \omega))$', rotation=270, fontsize=fontsize, labelpad=15)
    #cbar.ax.tick_params(labelsize=fontsize)

    plt.savefig('spectralGS.png', format='png', bbox_inches='tight', dpi = 600)
    #plt.tight_layout()
    #plt.show()


def plotPtGS(ptGS, eta):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    bins = np.arange(len(ptGS))
    ax.bar(bins, np.abs(ptGS), log=True, color='wheat')
    ax.hlines(1., -2., len(ptGS) + 1, linestyles='--', colors='gray')
    ax.set_xlim(-1, 31)
    ax.set_ylim(1e-12, 1e1)
    labelString = "g = {:.2f} \n $\omega$ = {:.2f}".format(eta, prms.w0)
    ax.text(20, 1e-4, labelString, fontsize=20)
    plt.show()


def plotPtGSWithCoh(ptGS, N, eta, T):
    #cohState = coherentState.getCoherentStateForN(N)
    ptGS = ptGS + 1e-16
    cohState = coherentState.getSqueezedState(eta, T)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    bins = np.arange(len(ptGS))
    ax.bar(bins, ptGS, log=False, color='wheat', label = "GS")
    ax.plot(cohState, color='red', linestyle = '', marker = 'x', label = "Squeezed state")
    #ax.hlines(1., -2., 30, linestyles='--', colors='gray')
    ax.set_xlim(-1, 11)
    #ax.set_ylim(1e-10, 9 * 1e1)
    labelString = "L = {:.0f} \n $\omega$ = {:.1f}".format(prms.chainLength, prms.w0)
    #ax.text(20, 1e-4, labelString, fontsize=20)
    plt.legend(loc = 'upper left')
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
        ax.plot(wVec, np.abs(np.imag(gfNonEqCohN0[0, :])), color=color, linestyle='-', label=labelStr)

        if (lInd == len(LArr) - 1):
            gWFloquet = spectralFunction.gLesserW(kVec, wVec, tAv, eta, cohN, damping)
            gWFloquetInt = 1. / (20. * tau) * (tAv[1] - tAv[0]) * np.sum(gWFloquet, axis=2)
            ax.plot(wVec, np.abs(np.imag(gWFloquetInt[0, :])), color='gray', linestyle='--', label='Floquet')

    ax.set_yscale('log')
    ax.set_yticks([1e0, 1e-1, 1e-2])
    ax.set_yticklabels(['$10^0$', '$10^{-1}$', '$10^{-2}$'])
    plt.legend()
    plt.show()


def greenWaterFall(kVec, wVec, gfNonEq, lArr, gfFloquet, eta):

    wVec = np.linspace(-4., 4., 2000, endpoint=False)
    kVec = np.linspace(-np.pi, np.pi, 17, endpoint=True)


    gfNonEq = np.abs(gfNonEq) + 1e-16
    gfFloquet = np.abs(gfFloquet) + 1e-16
    fig, ax = plt.subplots(nrows=1, ncols=2, gridspec_kw={'width_ratios': [1, 20]})
    fig.set_size_inches(16./2., 9./2.)
    ax[0].tick_params(left=False, labelleft=False, bottom=False, labelbottom = False)
    ax[0].axis('off')
    ax = ax[1]

    left, bottom, width, height = [0.125, 0.625, 0.3, 0.3]
    axIn1 = fig.add_axes([left, bottom, width, height])

    for axis in ['top', 'bottom', 'left', 'right']:
        axIn1.spines[axis].set_linewidth(1.5)

    left, bottom, width, height = [0.74, 0.625, 0.3, 0.3]
    axIn2 = fig.add_axes([left, bottom, width, height])

    for axis in ['top', 'bottom', 'left', 'right']:
        axIn2.spines[axis].set_linewidth(1.5)



    for kInd, kVal in enumerate(kVec):
        kInd = len(kVec) - kInd - 1
        offSet = 1. * 10**(0.4 * kInd)
        quantumPlot = np.flip(gfNonEq, axis = 2) * offSet
        floquetPlot = np.flip(gfFloquet, axis = 1) * offSet

        #plot quantum
        for lInd, lVal in enumerate(lArr):
            if (lInd == 0):
                color = "tan"
                labelString = "g = {:.1f} \n$\langle N_{{ph}} \\rangle$ = {:.0f}".format(0.1, 20)
            else:
                color = "lightsteelblue"
                labelString = "g = {:.1f} \n$\langle N_{{ph}} \\rangle$ = {:.1f}".format(1.0, 0.2)


            if (kInd == 0):
                ax.plot(wVec, quantumPlot[lInd, kInd, :], marker='', color=color, linestyle='-', linewidth = 3., label=labelString)
                axIn1.plot(wVec, quantumPlot[lInd, kInd, :], marker='', color=color, linestyle='-', linewidth = 3.)
                axIn2.plot(wVec, quantumPlot[lInd, kInd, :], marker='', color=color, linestyle='-', linewidth = 3.)
            else:
                ax.plot(wVec, quantumPlot[lInd, kInd, :], marker='', color=color, linestyle='-', linewidth = 3.)
                axIn1.plot(wVec, quantumPlot[lInd, kInd, :], marker='', color=color, linestyle='-', linewidth = 3.)
                axIn2.plot(wVec, quantumPlot[lInd, kInd, :], marker='', color=color, linestyle='-', linewidth = 3.)

        #plot floquet
        if (kInd == 0):
            ax.plot(wVec, floquetPlot[kInd, :], marker='', color='black', linestyle='-', linewidth = 0.5, label="Floquet")
            axIn1.plot(wVec, floquetPlot[kInd, :], marker='', color='black', linestyle='-', linewidth = 0.5)
            axIn2.plot(wVec, floquetPlot[kInd, :], marker='', color='black', linestyle='-', linewidth = 0.5)
        else:
            ax.plot(wVec, floquetPlot[kInd, :], marker='', color='black', linestyle='-', linewidth = 0.5)
            axIn1.plot(wVec, floquetPlot[kInd, :], marker='', color='black', linestyle='-', linewidth = 0.5)
            axIn2.plot(wVec, floquetPlot[kInd, :], marker='', color='black', linestyle='-', linewidth = 0.5)

    ax.set_yscale('log')
    ax.set_yticks([])
    ax.set_yticklabels([])
    ax.set_xticks([-4, -2, 0, 2, 4])
    ax.set_xticklabels(['-4', '-2', '0', '2', '4'])
    ax.set_xlabel('$\omega / t_h$', fontsize = fontsize + 4)
    ax.set_ylabel('$A(k, \omega)$', fontsize = fontsize + 4)
    ax.tick_params(axis='both', which='major', labelsize=fontsize)
    legend = ax.legend(fontsize = fontsize, loc = 'lower right', bbox_to_anchor=(1.225, 0.), edgecolor = 'black')
    legend.get_frame().set_alpha(None)
    legend.get_frame().set_boxstyle('Square', pad=0.2)
    legend.get_frame().set_linewidth(1.5)

    #arrow = patches.Arrow(-2, 1000, 2, 0, zorder = 100, width = 2)
    arrow = patches.FancyArrowPatch((-2.8, 35), (-0.8, 35), arrowstyle='<->', mutation_scale=20, zorder = 100, linewidth=2., color = 'black')
    ax.add_patch(arrow)
    ax.text(-1.95, 45, "$\Omega$", fontsize = fontsize + 4)

    rect = patches.Rectangle((-2.6, 2.), 1.5, 12.5, linewidth=1.5, edgecolor='black', facecolor='none', zorder = 100)
    ax.add_patch(rect)

    axIn1.set_ylim(2., 2. + 12.5)
    axIn1.set_yscale('log', subsy = [0])
    axIn1.set_yticks([])
    axIn1.set_xlim(-2.65, -2.65 + 1.5)
    axIn1.set_xticks([])
    axIn1.set_xticklabels([])
    axIn1.tick_params(axis='x', which='major', labelsize=14, width = 1.)

    rect = patches.Rectangle((.5, 1500), 1.5, 8000, linewidth=1.5, edgecolor='black', facecolor='none', zorder = 100)
    ax.add_patch(rect)

    axIn2.set_ylim(1500., 1500. + 8000)
    axIn2.set_yscale('log', subsy = [0])
    axIn2.set_yticks([])
    axIn2.set_xlim(.5, .5 + 1.5)
    axIn2.set_xticks([])
    axIn2.set_xticklabels([])
    axIn2.tick_params(axis='x', which='major', labelsize=14, width = 1.)


    #plt.savefig('waterfallWithInsets.pdf', format='pdf', bbox_inches='tight')
    plt.savefig('waterfallWithInsetsNew.png', format='png', bbox_inches='tight', dpi = 600)
    #plt.show()


def greenWaterFallOnlyFloquet(kVec, wVec, gfFloquet):
    gfFloquet = np.abs(gfFloquet) + 1e-16
    fig, ax = plt.subplots(nrows=1, ncols=1)
    fig.set_size_inches(16./2., 9./2.)
    ax.tick_params(left=False, labelleft=True, bottom=True, labelbottom = True)
    #ax.axis('off')

    for kInd, kVal in enumerate(kVec):
        kInd = len(kVec) - kInd - 1
        offSet = 1. * 10**(0.4 * kInd)
        floquetPlot = np.flip(gfFloquet, axis = 1) * offSet

        #plot floquet
        if (kInd == 0):
            ax.plot(wVec, floquetPlot[kInd, :], marker='', color='black', linestyle='-', linewidth = 0.5, label="Floquet")
        else:
            ax.plot(wVec, floquetPlot[kInd, :], marker='', color='black', linestyle='-', linewidth = 0.5)

    ax.set_yscale('log')
    ax.set_yticks([])
    ax.set_yticklabels([])
    ax.set_xticks([-6, -4, -2, 0, 2, 4, 6])
    ax.set_xticklabels(['-6', '-4', '-2', '0', '2', '4', '6'])
    ax.set_xlabel('$\omega / t_h$', fontsize = fontsize + 4)
    ax.set_ylabel('$A(k, \omega)$', fontsize = fontsize + 4)
    ax.tick_params(axis='both', which='major', labelsize=fontsize)
    #legend = ax.legend(fontsize = fontsize, loc = 'lower right', bbox_to_anchor=(1.225, 0.), edgecolor = 'black')
    #legend.get_frame().set_alpha(None)
    #legend.get_frame().set_boxstyle('Square', pad=0.2)
    #legend.get_frame().set_linewidth(1.5)

    #arrow = patches.Arrow(-2, 1000, 2, 0, zorder = 100, width = 2)
    #arrow = patches.FancyArrowPatch((-2.8, 35), (-0.8, 35), arrowstyle='<->', mutation_scale=20, zorder = 100, linewidth=2., color = 'black')
    #ax.add_patch(arrow)
    #ax.text(-1.95, 45, "$\Omega$", fontsize = fontsize + 4)


    plt.savefig('waterfallFloquet2.pdf', format='pdf', bbox_inches='tight')
    #plt.show()


def plotAnalyticalConductivity(eta1, eta2, eta3):

    gsT = - 2. / np.pi * prms.chainLength

    fac = np.sqrt(1 - 2. * eta1 * eta1 / (prms.w0) * gsT)

    omegaVec = np.linspace(-60 , 60, 4000000 , endpoint=True)
    delta = 0.005

    cond1 = calcConductivity(omegaVec, delta, eta1)
    cond2 = calcConductivity(omegaVec, delta, eta2)
    cond3 = calcConductivity(omegaVec, delta, eta3)

    cond1 = cond1 / np.amax(np.real(cond3))
    cond2 = cond2 / np.amax(np.real(cond3))
    cond3 = cond3 / np.amax(np.real(cond3))

    sum1 = np.sum(np.real(cond1))
    sum2 = np.sum(np.real(cond2))
    print("sum1 = {}".format(sum1))
    print("sum2 = {}".format(sum2))

    etas = np.linspace(0., 2., 20) * 1. / np.sqrt(prms.chainLength)
    etasLabels = np.linspace(0., 2., 20)
    gsKinetics = -coherentState.gsEffectiveKineticEnergyArray(etas)

    intConductivities = integratedConductivityArr(omegaVec, delta, etas)

    fig = plt.figure()
    fig.set_size_inches(0.75 * 16. / 4., 0.75 * 12 / 4.)

    ax = fig.add_subplot(111)

    left, bottom, width, height = [0.675, 0.41, 0.4, 0.3]
    axIn1 = fig.add_axes([left, bottom, width, height])

    left, bottom, width, height = [0.675, 0.85, 0.4, 0.3]
    axIn2 = fig.add_axes([left, bottom, width, height])

    ax.plot(omegaVec, np.real(cond2), color = 'lightsteelblue', linewidth = 1.5, linestyle = '-', label = "g = {}".format(0.1))
    ax.plot(omegaVec, np.real(cond1), color = 'tan', linewidth = 1.5, linestyle = '-', label = "g = {}".format(1.))
    ax.plot(omegaVec, np.real(cond3), color = 'black', linewidth = .5, linestyle = '-', label = "g = {}".format(0))


    #ax.set_ylim(1e-6, 1e4)
    #ax.set_yscale('log')
    ax.set_xlim(- 1.5 * prms.w0 * fac , 1.5 * prms.w0 * fac)

    #ax.set_yticks([1e2, 1e0, 1e-2, 1e-4, 1e-6])
    ax.set_xticks([-2., -1., 0., 1., 2.])
    ax.set_xticklabels([r'$-2$', r'$-1$', r'$0$', r'$1$', r'$2$', ])
    ax.tick_params(axis='both', which='major', labelsize=fontsize)

    ax.set_xlabel('$\omega$', fontsize=fontsize)
    ax.set_ylabel('Re$(\sigma)$', fontsize=fontsize)

    axIn1.plot(omegaVec, np.real(cond1), color = 'tan', linewidth = 1.5, linestyle = '-', label = "g = {}".format(1.))
    axIn1.plot(omegaVec, np.real(cond2), color = 'lightsteelblue', linewidth = 1.5, linestyle = '-', label = "g = {}".format(0.1))
    axIn1.plot(omegaVec, np.real(cond3), color = 'black', linewidth = .5, linestyle = '-', label = "g = {}".format(0))

    axIn1.set_ylim(0.979, 1.001)
    axIn1.set_xlim(-0.001, 0.001)

    axIn1.set_yticks([0.98, 1.])
    axIn1.set_xticks([])
    #axIn1.set_xticklabels([r'$-2$', r'$-1$', r'$0$', r'$1$', r'$2$', ])
    axIn1.tick_params(axis='both', which='major', labelsize=fontsize)

    axIn2.plot(etasLabels, gsKinetics, color = 'black', linewidth = 1.)
    axIn2.plot(etasLabels, 1. / np.pi * intConductivities, color = 'red', linewidth = 1., linestyle = '--')
    yLimBot = np.amin(gsKinetics) - 0.001
    yLimTop = np.amax(gsKinetics) + 0.001
    axIn2.set_ylim(yLimBot, yLimTop)
    axIn2.set_yticks([0.625, 0.635])
    axIn2.set_xticks([0, 2])
    axIn2.set_xticklabels(['$g = 0$', '$g = 2$'])
    axIn2.vlines(0., yLimBot, yLimTop, color = 'black', linestyle = 'dotted', linewidth = 1.)
    axIn2.vlines(0.1, yLimBot, yLimTop, color = 'lightsteelblue', linestyle = 'dotted', linewidth = 1.)
    axIn2.vlines(1., yLimBot, yLimTop, color = 'tan', linestyle = 'dotted', linewidth = 1.)
    axIn2.tick_params(axis='both', which='major', labelsize=fontsize)

    #axIn2.set_xlabel('$g$', fontsize=fontsize)
    axIn2.set_ylabel(r'$-e_{\mathrm{kin}}$', fontsize=fontsize)


    legend = ax.legend(fontsize = fontsize, loc = 'upper center', bbox_to_anchor=(.25, 1.05), edgecolor = 'black', ncol = 1)
    legend.get_frame().set_alpha(0.0)
    legend.get_frame().set_boxstyle('Square', pad=0.0)
    legend.get_frame().set_linewidth(0)

    plt.savefig('conductivityGS.png', format='png', bbox_inches='tight', dpi = 600)
    #plt.tight_layout()
    #plt.show()


def plotAnalyticalConductivityImaginary(eta1, eta2, eta3):

    gsT = - 2. / np.pi * prms.chainLength

    fac = np.sqrt(1 - 2. * eta1 * eta1 / (prms.w0) * gsT)

    omegaVec = np.linspace(-60 , 60, 4000000 , endpoint=True)
    delta = 0.005

    cond1 = calcConductivity(omegaVec, delta, eta1)
    cond2 = calcConductivity(omegaVec, delta, eta2)
    cond3 = calcConductivity(omegaVec, delta, eta3)

    cond1 = cond1 / np.amax(np.imag(cond3))
    cond2 = cond2 / np.amax(np.imag(cond3))
    cond3 = cond3 / np.amax(np.imag(cond3))

    etas = np.linspace(0., 2., 20) * 1. / np.sqrt(prms.chainLength)

    fig = plt.figure()
    fig.set_size_inches(0.5 * 16. / 4., 0.5 * 12 / 4.)

    ax = fig.add_subplot(111)

    left, bottom, width, height = [0.16, 0.65, 0.3, 0.2]
    axIn1 = fig.add_axes([left, bottom, width, height])

    left, bottom, width, height = [0.675, 0.14, 0.2, 0.2]
    axIn2 = fig.add_axes([left, bottom, width, height])

    ax.plot(omegaVec, np.imag(cond1), color = 'tan', linewidth = 1.5, linestyle = '-', label = "g = {}".format(1.))
    ax.plot(omegaVec, np.imag(cond2), color = 'lightsteelblue', linewidth = 1.5, linestyle = '-', label = "g = {}".format(0.1))
    ax.plot(omegaVec, np.imag(cond3), color = 'black', linewidth = .5, linestyle = '-', label = "g = {}".format(0))


    #ax.set_ylim(1e-6, 1e4)
    #ax.set_yscale('log')
    ax.set_xlim(- 1.5 * prms.w0 * fac , 1.5 * prms.w0 * fac)

    #ax.set_yticks([1e2, 1e0, 1e-2, 1e-4, 1e-6])
    ax.set_xticks([-2., -1., 0., 1., 2.])
    ax.set_xticklabels([r'$-2$', r'$-1$', r'$0$', r'$1$', r'$2$', ])
    ax.tick_params(axis='both', which='major', labelsize=fontsize)

    ax.set_xlabel('$\omega$', fontsize=fontsize)
    ax.set_ylabel('Im$(\sigma)$', fontsize=fontsize)

    axIn1.plot(omegaVec, np.imag(cond1), color = 'tan', linewidth = 1.5, linestyle = '-', label = "g = {}".format(1.))
    axIn1.plot(omegaVec, np.imag(cond2), color = 'lightsteelblue', linewidth = 1.5, linestyle = '-', label = "g = {}".format(0.1))
    axIn1.plot(omegaVec, np.imag(cond3), color = 'black', linewidth = .5, linestyle = '-', label = "g = {}".format(0))

    xlimBot = -1.15
    xlimTop = -0.85
    ylimBot = -0.018
    ylimTop = -0.002
    axIn1.set_xlim(xlimBot, xlimTop)
    axIn1.set_ylim(ylimBot, ylimTop)
    axIn1.set_xticks([])
    axIn1.set_yticks([])

    rect = patches.Rectangle((xlimBot, -0.04), xlimTop - xlimBot, 0.07, linewidth=.4, edgecolor='black', facecolor='none', zorder = 100)
    ax.add_patch(rect)
    ax.plot([xlimBot, -2.055], [0.03, 0.425], linewidth = .5, color = 'black')
    ax.plot([xlimTop, -0.3], [0.03, 0.425], linewidth = .5, color = 'black')

    axIn2.plot(omegaVec, np.imag(cond1), color = 'tan', linewidth = 1.5, linestyle = '-', label = "g = {}".format(1.))
    axIn2.plot(omegaVec, np.imag(cond2), color = 'lightsteelblue', linewidth = 1.5, linestyle = '-', label = "g = {}".format(0.1))
    axIn2.plot(omegaVec, np.imag(cond3), color = 'black', linewidth = .5, linestyle = '-', label = "g = {}".format(0))

    xlimBot = -0.007
    xlimTop = -0.003
    ylimBot = -1.001
    ylimTop = -0.979
    axIn2.set_xlim(xlimBot, xlimTop)
    axIn2.set_ylim(ylimBot, ylimTop)
    axIn2.set_xticks([])
    axIn2.set_yticks([-1., -0.98])
    axIn2.set_yticklabels(['${-}1.0$', '${-}0.98$'])
    axIn2.tick_params(axis='both', which='major', labelsize=fontsize - 4, pad = 0)

    #rect = patches.Rectangle((xlimBot, ylimBot), xlimTop - xlimBot, ylimTop - ylimBot, linewidth=.4, edgecolor='black', facecolor='none', zorder = 100)
    rect = patches.Rectangle((-0.06, -1.025), 0.1, 0.1, linewidth=.4, edgecolor='black', facecolor='none', zorder = 100)
    ax.add_patch(rect)

    legend = ax.legend(fontsize = fontsize, loc = 'upper center', bbox_to_anchor=(.9, 1.3), edgecolor = 'black', ncol = 1)
    legend.get_frame().set_alpha(1.0)
    legend.get_frame().set_boxstyle('Square', pad=0.1)
    #legend.get_frame().set_linewidth(0)

    plt.savefig('conductivityGSImaginary.png', format='png', bbox_inches='tight', dpi = 600)
    #plt.tight_layout()
    #plt.show()


def calcConductivity(omegaVec, delta, eta):
    gsT = - 2. / np.pi * prms.chainLength

    fac = np.sqrt(1 - 2. * eta * eta / (prms.w0) * gsT)


    drudePart = -1j / (omegaVec + 1j * delta) * gsT / prms.chainLength * (1 - eta ** 2 / (2.) * fac )

    cavityFreqPart = 1j / (omegaVec - fac * prms.w0 + 1j * delta) - 1j / (omegaVec + fac * prms.w0 + 1j * delta)
    cavityPart = eta ** 2 * gsT ** 2 / prms.chainLength * 1. / fac * 1. / (omegaVec + 1j * delta) * cavityFreqPart

    cond = drudePart + cavityPart
    return cond

def integratedConductivityArr(omegaVec, delta, etas):
    intConductivity = np.zeros(len(etas))
    for indEta, eta in enumerate(etas):
        condTemp = np.real(calcConductivity(omegaVec, delta, eta))
        condInt =  (omegaVec[1] - omegaVec[0]) * np.sum(condTemp[0 : -1]) / prms.chainLength
        intConductivity[indEta] = condInt
    return intConductivity * prms.chainLength

def plotLandscapesAllOrders(etas, orderH):
    orderH = 3
    bins = 500

    #landscapes = fsShift.getManyEnergyLandscapes(etas, orderH, bins)
    #file = h5py.File("data/landscapes" + str(orderH) + ".h5", 'w')
    #file.create_dataset("landscapes", data=landscapes)
    #file.create_dataset("etas", data=etas)
    #file.close()

    file = h5py.File("data/landscapes" + str(orderH) + ".h5", 'r')
    landscapes = file["landscapes"][()]
    etas = file["etas"][()]
    file.close()

    Ls = np.logspace(1., 4., 31, endpoint = True)
    etasNonNorm = etas * np.sqrt(prms.chainLength)

    #photonOccs2 = fsShift.occupationsForLengthsZeroShift(Ls, etasNonNorm, 2)
    #photonOccsArb = fsShift.occupationsForLengthsZeroShift(Ls, etasNonNorm, 3)
    #file = h5py.File("data/occsAll.h5", 'w')
    #file.create_dataset("occs2", data=photonOccs2)
    #file.create_dataset("occs3", data=photonOccsArb)
    #file.close()

    file = h5py.File("data/occsAll.h5", 'r')
    photonOccs2 = file["occs2"][()]
    photonOccsArb = file["occs3"][()]
    file.close()

    fig = plt.figure()
    fig.set_size_inches(0.6 * 16. / 4., 0.6 * 12 / 4.)
    ax = fig.add_subplot(111)

    left, bottom, width, height = [0.725, 0.55, 0.4, 0.4]
    axIn1 = fig.add_axes([left, bottom, width, height])

    for axis in ['top', 'bottom', 'left', 'right']:
        ax.spines[axis].set_linewidth(0.5)
        axIn1.spines[axis].set_linewidth(0.5)

    cmap = plt.cm.get_cmap('gist_earth')
    xArr = np.linspace(0., 2. * np.pi, bins)
    for indEta in range(len(etas)):
        eta = etas[indEta]
        etaLabel = eta * np.sqrt(prms.chainLength)
        color = cmap(etaLabel / (etas[-1] * np.sqrt(prms.chainLength) + 0.1))
        ax.plot(xArr, landscapes[indEta, :] / prms.chainLength, color = color, label = r'g = {:.2f}'.format(etaLabel), linewidth = 1.5)

        axIn1.plot(Ls, photonOccsArb[:, indEta], color = color, linewidth = 1.)
        axIn1.plot(Ls, photonOccs2[:, indEta], color = 'black', linestyle = 'dotted', linewidth = 1.)

    #labelString = "$\omega$ = {:.2f}".format(prms.w0)
    #ax.text(0., .5, labelString, fontsize = 14)
    ax.set_ylim(-.7, 1.2)
    ax.set_ylabel("$e[t]$", fontsize = 10, labelpad = -2)
    ax.set_xlabel("$\mathrm{FS}$ $\mathrm{center}$", fontsize = 10)

    ax.set_xticks([0., np.pi / 2., np.pi, 1.5 * np.pi, 2. * np.pi])
    ax.set_xticklabels(['0', r'$\frac{\pi}{2}$', '$\pi$', r'$\frac{3\pi}{2}$', '$2 \pi$'], fontsize = fontsize)

    yLimBot = -0.02
    yLimTop = 0.23
    axIn1.set_ylim(yLimBot, yLimTop)
    axIn1.vlines(1010, yLimBot, yLimTop, color = 'red', linestyle = '-', linewidth = .4)

    axIn1.set_xscale('log')
    axIn1.set_xlabel('$\log(L)$', fontsize = 8)
    axIn1.set_ylabel('$N_{\mathrm{pt}}$', fontsize = 8)

    axIn1.set_xticks([1e2, 1e4])
    axIn1.set_xticklabels(['$10^2$', '$10^4$'], fontsize = 8)
    axIn1.set_yticks([0.0, 0.2])
    axIn1.set_yticklabels(['$0.0$', '$0.2$'], fontsize = 8)

    legend = ax.legend(fontsize = fontsize - 4, loc = 'upper left', bbox_to_anchor=(0., 1.12), edgecolor = 'black', ncol = 1)
    legend.get_frame().set_alpha(0.95)
    legend.get_frame().set_boxstyle('Square', pad=0.1)
    legend.get_frame().set_linewidth(0.5)

    plt.savefig('fsShiftsAllOrders.png', format='png', bbox_inches='tight', dpi = 600)
    #plt.tight_layout()
    #plt.show()

def plotLandscapes1Order(etas, orderH):
    orderH = 1
    bins = 100

    #landscapes = fsShift.getManyEnergyLandscapes(etas, orderH, bins)
    #file = h5py.File("data/landscapes" + str(orderH) + ".h5", 'w')
    #file.create_dataset("landscapes", data=landscapes)
    #file.create_dataset("etas", data=etas)
    #file.close()

    file = h5py.File("data/landscapes" + str(orderH) + ".h5", 'r')
    landscapes = file["landscapes"][()]
    etas = file["etas"][()]
    file.close()

    Ls = np.logspace(1., 4., 11, endpoint = True)
    etasNonNorm = etas * np.sqrt(prms.chainLength)

    #photonOccs1 = fsShift.occupationsForLengths(Ls, etasNonNorm, 1, 500)
    #file = h5py.File("data/occsOne.h5", 'w')
    #file.create_dataset("occs", data=photonOccs1)
    #file.close()

    file = h5py.File("data/occsOne.h5", 'r')
    photonOccs1 = file["occs"][()]
    file.close()

    fig = plt.figure()
    fig.set_size_inches(0.6 * 16. / 4., 0.6 * 12 / 4.)
    ax = fig.add_subplot(111)

    left, bottom, width, height = [0.8, 0.65, 0.4, 0.3]
    axInTop = fig.add_axes([left, bottom, width, height])

    left, bottom, width, height = [0.8, 0.5, 0.4, 0.1]
    axInBot = fig.add_axes([left, bottom, width, height])

    left, bottom, width, height = [0.8, 0.6, 0.4, 0.05]
    axInMid = fig.add_axes([left, bottom, width, height])

    for axis in ['top', 'bottom', 'left', 'right']:
        ax.spines[axis].set_linewidth(0.5)
        axInTop.spines[axis].set_linewidth(0.5)
        axInBot.spines[axis].set_linewidth(0.5)


    cmap = plt.cm.get_cmap('gist_earth')
    xArr = np.linspace(0., 2. * np.pi, bins)
    for indEta in range(len(etas)):
        eta = etas[indEta]
        etaLabel = eta * np.sqrt(prms.chainLength)
        color = cmap(etaLabel / (etas[-1] * np.sqrt(prms.chainLength) + 0.1))
        ax.plot(xArr, landscapes[indEta, :] / prms.chainLength, color = color, label = r'$g = {:.2f}$'.format(etaLabel), linewidth = 1.5)

        if(etaLabel > 0.8):
            axInTop.loglog(Ls, photonOccs1[:, indEta], color = color, linewidth = 1.)
        else:
            if(indEta == 2):
                axInBot.plot(Ls, photonOccs1[:, indEta], color = color, linewidth = 1., linestyle = '--')
            else:
                axInBot.plot(Ls, photonOccs1[:, indEta], color = color, linewidth = 1., linestyle = '-')

    axInTop.loglog(Ls, Ls, color = 'black', linewidth = 1.2, linestyle = '--', label = "$L$")



    #labelString = "$\omega$ = {:.2f}".format(prms.w0)
    #ax.text(0., .5, labelString, fontsize = 14)
    ax.set_ylim(-1.8, 1.2)
    ax.set_ylabel("$e[t]$", fontsize = 10, labelpad = -2)
    ax.set_xlabel("$\mathrm{FS}$ $\mathrm{center}$", fontsize = 10)

    ax.set_xticks([0., np.pi / 2., np.pi, 1.5 * np.pi, 2. * np.pi])
    ax.set_xticklabels(['0', r'$\frac{\pi}{2}$', '$\pi$', r'$\frac{3\pi}{2}$', '$2 \pi$'], fontsize = fontsize)

    axInBot.set_xscale('log')
    axInMid.set_xscale('log')
    axInBot.set_xlabel('$\log(L)$', fontsize = 7)
    axInTop.set_ylabel('$\log (N_{\mathrm{pt}})$', fontsize = 7, rotation = 0)
    axInTop.yaxis.set_label_coords(-0.24, .8)

    axInTop.set_xticks([])
    axInBot.set_yticks([0])
    axInMid.set_xticks([])
    axInMid.set_yticks(())
    #axInBot.set_yticklabels(['$N_{\mathrm{pt}} = 0$'], fontsize = 8)
    axInBot.set_yticklabels(['$0$'], fontsize = 8)

    axInBot.set_xticks([1e2, 1e4])
    axInBot.set_xticklabels(['$10^2$', '$10^4$'], fontsize = 8)

    axInTop.set_yticks([1e0, 1e2])
    axInTop.set_yticklabels(['$10^0$', '$10^2$'], fontsize = 8)

    yLimBot = 5. * 1e-2
    yLimTop = 3. * 1e4
    axInTop.set_ylim(yLimBot, yLimTop)
    axInTop.vlines(1010, yLimBot, yLimTop, color = 'red', linestyle = '-', linewidth = .4)

    yLimBot2 = -0.1
    yLimTop2 = 0.1
    axInBot.set_ylim(yLimBot2, yLimTop2)
    axInBot.vlines(1010, yLimBot2, yLimTop2, color = 'red', linestyle = '-', linewidth = .4)

    yLimBot3 = 0
    yLimTop3 = 1
    axInMid.set_ylim(yLimBot3, yLimTop3)
    axInMid.vlines(1010, yLimBot3, yLimTop3, color = 'red', linestyle = '-', linewidth = .4)

    axInTop.set_xlim(8 * 1e0, 1.5 * 1e4)
    axInBot.set_xlim(8 * 1e0, 1.5 * 1e4)
    axInMid.set_xlim(8 * 1e0, 1.5 * 1e4)

    axInTop.spines['bottom'].set_visible(False)
    axInBot.spines['top'].set_visible(False)
    for axis in ['top', 'bottom', 'left', 'right']:
        axInMid.spines[axis].set_visible(False)

    #ax.spines['right'].set_visible(False)

    legend = ax.legend(fontsize = fontsize - 4, loc = 'upper left', bbox_to_anchor=(0., 1.12), edgecolor = 'black', ncol = 1)
    legend.get_frame().set_alpha(0.95)
    legend.get_frame().set_boxstyle('Square', pad=0.1)
    legend.get_frame().set_linewidth(0.5)

    legend = axInTop.legend(fontsize = fontsize - 4, loc = 'upper left', bbox_to_anchor=(0., 1.0), edgecolor = 'black', ncol = 1)
    legend.get_frame().set_alpha(1.0)
    legend.get_frame().set_boxstyle('Square', pad=0.1)
    legend.get_frame().set_linewidth(0.0)

    plt.savefig('fsShifts1.png', format='png', bbox_inches='tight', dpi = 600)
    #plt.tight_layout()
    #plt.show()


def plotLandscapes2Order(etas, orderH):
    orderH = 2
    bins = 100

    #landscapes = fsShift.getManyEnergyLandscapes(etas, orderH, bins)
    #file = h5py.File("data/landscapes" + str(orderH) + ".h5", 'w')
    #file.create_dataset("landscapes", data=landscapes)
    #file.create_dataset("etas", data=etas)
    #file.close()

    file = h5py.File("data/landscapes" + str(orderH) + ".h5", 'r')
    landscapes = file["landscapes"][()]
    etas = file["etas"][()]
    file.close()

    Ls = np.logspace(1., 4., 11, endpoint = True)
    etasNonNorm = etas * np.sqrt(prms.chainLength)

    fig = plt.figure()
    fig.set_size_inches(0.6 * 16. / 4., 0.6 * 12 / 4.)
    ax = fig.add_subplot(111)

    for axis in ['top', 'bottom', 'left', 'right']:
        ax.spines[axis].set_linewidth(0.5)

    cmap = plt.cm.get_cmap('gist_earth')
    xArr = np.linspace(0., 2. * np.pi, bins)
    for indEta in range(len(etas)):
        eta = etas[indEta]
        etaLabel = eta * np.sqrt(prms.chainLength)
        color = cmap(etaLabel / (etas[-1] * np.sqrt(prms.chainLength) + 0.1))
        ax.plot(xArr, landscapes[indEta, :] / prms.chainLength, color = color, label = r'$g = {:.2f}$'.format(etaLabel), linewidth = 1.5)


    #labelString = "$\omega$ = {:.2f}".format(prms.w0)
    #ax.text(0., .5, labelString, fontsize = 14)
    ax.set_ylim(-1.8, 1.2)
    ax.set_ylabel("$e[t]$", fontsize = 10, labelpad = -2)
    ax.set_xlabel("$\mathrm{FS}$ $\mathrm{center}$", fontsize = 10)

    ax.set_xticks([0., np.pi / 2., np.pi, 1.5 * np.pi, 2. * np.pi])
    ax.set_xticklabels(['0', r'$\frac{\pi}{2}$', '$\pi$', r'$\frac{3\pi}{2}$', '$2 \pi$'], fontsize = fontsize)

    #ax.spines['right'].set_visible(False)

    legend = ax.legend(fontsize = fontsize - 4, loc = 'upper left', bbox_to_anchor=(0., 1.3), edgecolor = 'black', ncol = 1)
    legend.get_frame().set_alpha(0.95)
    legend.get_frame().set_boxstyle('Square', pad=0.1)
    legend.get_frame().set_linewidth(0.5)

    plt.savefig('fsShifts2.png', format='png', bbox_inches='tight', dpi = 600)
    #plt.tight_layout()
    #plt.show()


def plotOccsLs(etasNonNorm, orderH):
    bins = 1000

    Ls = np.logspace(1., 7., 11, endpoint = True)
    occs = fsShift.occupationsForLengths(Ls, etasNonNorm, orderH, bins) + 1e-7
    file = h5py.File("data/sclaing_order" + str(orderH) + "h5", 'w')
    file.create_dataset("occs", data=occs)
    file.create_dataset("etasNonNorm", data=etasNonNorm)
    file.create_dataset("Ls", data=Ls)
    file.close()

    #file = h5py.File("data/sclaing_order" + str(orderH) + ".h5", 'r')
    #occs = file["occs"][()]
    #etasNonNorm = file["etasNonNorm"][()]
    #Ls = file["Ls"][()]


    fig, ax = plt.subplots(nrows=1, ncols=1)
    cmap = plt.cm.get_cmap('gist_earth')
    #ax.plot(etasNonNorm, occs[0, :])
    #ax.plot(etasNonNorm, occs[1, :])
    for indEta, eta in enumerate(etasNonNorm):
        color = cmap(eta / (etasNonNorm[-1] + 0.1))
        ax.loglog(Ls, occs[:, indEta], color=color, label=r'g = {:.2f}'.format(eta))
    plt.legend()
    plt.show()



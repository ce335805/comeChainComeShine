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

mpl.rcParams['font.family'] = 'sans-serif'
mpl.rcParams['lines.linewidth'] = 3
mpl.rcParams['lines.markersize'] = 10
mpl.rcParams['font.size'] = 20  # <-- change fonsize globally
mpl.rcParams['legend.fontsize'] = 14
mpl.rcParams['axes.titlesize'] = 20
mpl.rcParams['axes.labelsize'] = 20
mpl.rcParams['xtick.major.size'] = 3
mpl.rcParams['ytick.major.size'] = 3
mpl.rcParams['xtick.major.width'] = .7
mpl.rcParams['ytick.major.width'] = .7
mpl.rcParams['xtick.direction'] = 'out'
mpl.rcParams['ytick.direction'] = 'out'
mpl.rcParams['figure.titlesize'] = 20
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
    fig.set_size_inches(16. / 2., 9. / 2.)

    ax = fig.add_subplot(111)

    left, bottom, width, height = [0.1275, 0.115, 0.615, 0.25]
    axIn1 = fig.add_axes([left, bottom, width, height])
    #axIn1.set_yscale('log')

    left, bottom, width, height = [0.25, 0.205, 0.15, 0.15]
    axIn2 = fig.add_axes([left, bottom, width, height])

    left, bottom, width, height = [0.425, 0.205, 0.15, 0.15]
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

    print("selfE = {}".format(selfE))

    ax.arrow(-np.pi/2, selfE, 0., prms.w0, length_includes_head = True, color = 'darkseagreen', width = 0.025, head_width = 0.11, head_length = 0.15)

    ax.set_ylim(-6, 4.5)

    ax.set_xticks([-np.pi / 2., 0., np.pi / 2., np.pi])
    ax.set_xticklabels([r'$-\frac{\pi}{2}$', r'0', r'$\frac{\pi}{2}$', r'$\pi$'])
    ax.tick_params(axis='both', which='major', labelsize=fontsize)

    ax.set_xlabel('$k$', fontsize=fontsize + 4)
    ax.set_ylabel('$\omega$', fontsize=fontsize + 4)


    axIn1.plot(-wVec, spec[:, -1], color = 'lightsteelblue', linewidth = 3)
    axIn1.plot(-wVec, spec[:, prms.chainLength//4], color = 'tan', linewidth = 3)

    xLimBot = -.4
    xLimTop = 2.5
    yLimBot = spec[-1, prms.chainLength //2] - 0.0001
    yLimTop = spec[-1, prms.chainLength //2] + .4 # + 0.01 for second shakeoff
    axIn1.set_xlim(xLimBot, xLimTop)
    axIn1.set_ylim(yLimBot, yLimTop)
    axIn1.set_yticks([])
    axIn1.set_xticks([])
    axIn1.vlines(2. * np.cos(np.pi / 2. + np.pi / prms.chainLength), yLimBot, yLimTop, color = 'black', linestyle = '--')
    axIn1.vlines(2. * np.cos(np.pi / 2. + np.pi / prms.chainLength) + selfE, yLimBot, yLimTop, color = 'tan', linestyle = '--')
    axIn1.vlines(2., yLimBot, yLimTop, color = 'black', linestyle = '--')
    axIn1.vlines(2. * dynLoc, yLimBot, yLimTop, color = 'lightsteelblue', linestyle = '--')

#    ax.vlines(np.pi - 0.05, xLimBot, xLimTop, color = 'lightsteelblue', linestyles='dotted', linewidth = 4)

    rect = patches.Rectangle((np.pi - .11, xLimBot), .1, xLimTop - xLimBot, linewidth=1.5, edgecolor='lightsteelblue', facecolor='none', zorder=100)
    ax.add_patch(rect)

#    ax.vlines(np.pi / 2., xLimBot, xLimTop, color = 'tan', linestyles='dotted', linewidth = 4)

    rect = patches.Rectangle((np.pi/2. - .05, xLimBot), .1, xLimTop - xLimBot, linewidth=1.5, edgecolor='tan', facecolor='none', zorder=100)
    ax.add_patch(rect)

    axIn2.plot(-wVec, spec[:, prms.chainLength//4], color = 'tan', linewidth = 1.5)
    xLimBot = -.08
    xLimTop = .02
    yLimBot = spec[-1, prms.chainLength //2] - 0.0001
    yLimTop = spec[-1, prms.chainLength //2] + 7. # + 0.01 for second shakeoff
    axIn2.set_xlim(xLimBot, xLimTop)
    axIn2.set_ylim(yLimBot, yLimTop)
    axIn2.vlines(2. * np.cos(np.pi / 2. + np.pi / prms.chainLength), yLimBot, yLimTop, color = 'black', linestyle = 'dotted', linewidth = 1.)
    axIn2.vlines(2. * np.cos(np.pi / 2. + np.pi / prms.chainLength) + selfE, yLimBot, yLimTop, color = 'tan', linestyle = 'dotted', linewidth = 1.)

    axIn2.set_yticks([])

    axIn3.plot(-wVec, spec[:, -1], color = 'lightsteelblue', linewidth = 1.5)
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
    axIn3.set_xticklabels([r'$1.97$', r'$2$', r'$2.03$'])

    cbar.ax.set_ylabel(r'log$(A(k, \omega))}$', rotation=270, fontsize=fontsize, labelpad=20)
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
    ax.bar(bins, np.abs(ptGS), log=True, color='wheat', label = "GS")
    ax.plot(np.abs(cohState), color='red', linestyle = '', marker = 'x', label = "Squeezed state")
    #ax.hlines(1., -2., 30, linestyles='--', colors='gray')
    ax.set_xlim(-1, 51)
    ax.set_ylim(1e-10, 9 * 1e1)
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

    ax.plot(omegaVec, np.real(cond1), color = 'tan', linewidth = 1.5, linestyle = '-', label = "g = {}".format(1.))
    ax.plot(omegaVec, np.real(cond2), color = 'lightsteelblue', linewidth = 1.5, linestyle = '-', label = "g = {}".format(0.1))
    ax.plot(omegaVec, np.real(cond3), color = 'black', linewidth = .5, linestyle = '-', label = "g = {}".format(0))


    ax.set_ylim(1e-6, 1e4)
    ax.set_yscale('log')
    ax.set_xlim(- 1.5 * prms.w0 * fac , 1.5 * prms.w0 * fac)

    ax.set_yticks([1e2, 1e0, 1e-2, 1e-4, 1e-6])
    ax.set_xticks([-2., -1., 0., 1., 2.])
    ax.set_xticklabels([r'$-2$', r'$-1$', r'$0$', r'$1$', r'$2$', ])
    ax.tick_params(axis='both', which='major', labelsize=fontsize)

    ax.set_xlabel('$\omega$', fontsize=fontsize)
    ax.set_ylabel('log$($Re$(\sigma))$', fontsize=fontsize)

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

    left, bottom, width, height = [0.16, 0.6, 0.3, 0.25]
    axIn1 = fig.add_axes([left, bottom, width, height])

    left, bottom, width, height = [0.675, 0.15, 0.2, 0.3]
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

    xlimBot = -1.75
    xlimTop = -1.
    ylimBot = -0.011
    ylimTop = -0.002
    axIn1.set_xlim(xlimBot, xlimTop)
    axIn1.set_ylim(ylimBot, ylimTop)
    axIn1.set_xticks([])
    axIn1.set_yticks([])

    rect = patches.Rectangle((xlimBot, -0.04), xlimTop - xlimBot, 0.07, linewidth=.4, edgecolor='black', facecolor='none', zorder = 100)
    ax.add_patch(rect)
    ax.plot([xlimBot, -2.055], [0.03, 0.3], linewidth = .5, color = 'black')
    ax.plot([xlimTop, -0.3], [0.03, 0.3], linewidth = .5, color = 'black')

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

    legend = ax.legend(fontsize = fontsize, loc = 'upper center', bbox_to_anchor=(.9, 1.2), edgecolor = 'black', ncol = 1)
    legend.get_frame().set_alpha(1.0)
    legend.get_frame().set_boxstyle('Square', pad=0.1)
    #legend.get_frame().set_linewidth(0)

    plt.savefig('conductivityGSImaginary.png', format='png', bbox_inches='tight', dpi = 600)
    #plt.tight_layout()
    #plt.show()


def calcConductivity(omegaVec, delta, eta):
    gsT = - 2. / np.pi * prms.chainLength

    fac = np.sqrt(1 - 2. * eta * eta / (prms.w0) * gsT)

    drudePart = -1j / (omegaVec + 1j * delta) * gsT * (1 - eta ** 2 / (2.) * fac)

    cavityFreqPart = 1j / (omegaVec - fac * prms.w0 + 1j * delta) - 1j / (omegaVec + fac * prms.w0 + 1j * delta)
    cavityPart = eta ** 2 * gsT ** 2 / prms.chainLength * fac * 1. / (omegaVec + 1j * delta) * cavityFreqPart

    cond = drudePart + cavityPart
    return cond

def integratedConductivityArr(omegaVec, delta, etas):
    intConductivity = np.zeros(len(etas))
    for indEta, eta in enumerate(etas):
        condTemp = np.real(calcConductivity(omegaVec, delta, eta))
        condInt =  (omegaVec[1] - omegaVec[0]) * np.sum(condTemp[0 : -1]) / prms.chainLength
        intConductivity[indEta] = condInt
    return intConductivity

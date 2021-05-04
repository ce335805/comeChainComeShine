import numpy as np
import matplotlib.pyplot as plt
import globalSystemParams as prms
from matplotlib import ticker
from matplotlib.colors import LogNorm
from coherentState import coherentState
from floquet import spectralFunction
from nonEqGreen import nonEqGreen
import matplotlib.patches as patches

fontsize = 14

def plotSpec(kVec, wVec, spec):
    spec = np.roll(spec, prms.chainLength // 2 - 1, axis=1)
    kVecPosNeg = np.linspace(np.pi, -np.pi, prms.chainLength, endpoint=False)
    kVecPosNeg = np.flip(kVecPosNeg)

    fig = plt.figure()
    ax = fig.add_subplot(111)
    CS = ax.contourf(kVecPosNeg, -wVec, spec, 500, cmap='gnuplot2_r')
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
    CS = ax.contourf(kVecPosNeg, -wVec, spec, cmap='pink', norm=LogNorm(), levels=lvls)
    cbar = fig.colorbar(CS, ax=ax)
    cbar.set_ticks([1e-0, 1e-1, 1e-2])
    cbar.set_ticklabels([r'$10^{0}$', r'$10^{-1}$', r'$10^{-2}$'])
    # ax.arrow(0., 2. * prms.t, 0., - 4. * prms.t, length_includes_head = True, color = 'white', width = 0.025, head_width = 0.11, head_length = 0.015)
    # ax.arrow(np.pi/2, 0., 0., prms.w0, length_includes_head = True, color = 'white', width = 0.025, head_width = 0.11, head_length = 0.015)
    plt.show()

    print('{:.3f}'.format(spec[0, 0]))


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
                color = "peru"
                labelString = "g = {:.1f} \n$\langle N_{{ph}} \\rangle$ = {:.0f}".format(0.1, 20)
            else:
                color = "lightskyblue"
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
    plt.savefig('waterfallWithInsets.png', format='png', bbox_inches='tight', dpi = 600)
    #plt.show()


def greenWaterFall(kVec, wVec, gfNonEq, lArr, gfFloquet, eta):
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
                color = "peru"
                labelString = "g = {:.1f} \n$\langle N_{{ph}} \\rangle$ = {:.0f}".format(0.1, 20)
            else:
                color = "lightskyblue"
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
    #plt.savefig('waterfallWithInsets.png', format='png', bbox_inches='tight', dpi = 600)
    plt.show()

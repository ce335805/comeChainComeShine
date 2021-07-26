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
from conductivity import calcConductivity
import matplotlib.cm as cm
import matplotlib.colors
from arb_order import photonState
from GiacomosPlot import gsSqueezing
import h5py


#mpl.use("pgf")
#pgf_with_pdflatex = {
#    "pgf.texsystem": "pdflatex",
#    "pgf.preamble": [
#        r"\usepackage[utf8x]{inputenc}",
#        r"\usepackage[T1]{fontenc}",
#        r"\usepackage{cmbright}",
#         ]
#}
#pgf_with_pdflatex = {
#    "pgf.texsystem": "pdflatex",
#    "pgf.preamble": [
#        r"\usepackage[utf8x]{inputenc}",
#        r"\usepackage[T1]{fontenc}",
#        r"\usepackage{cmbright}",
#        r"\renewcommand{\familydefault}{\sfdefault}",
#        r"\usepackage[scaled=1]{helvet}",
#        r"\usepackage[helvet]{sfmath}",
#        r"\everymath={\sf}"
#         ]
#}
#mpl.rcParams.update(pgf_with_pdflatex)




mpl.rcParams['font.family'] = 'Helvetica'
mpl.rcParams['lines.linewidth'] = 2
mpl.rcParams['lines.markersize'] = 8
mpl.rcParams['font.size'] = 8  # <-- change fonsize globally
mpl.rcParams['legend.fontsize'] = 8
mpl.rcParams['axes.titlesize'] = 8
mpl.rcParams['axes.labelsize'] = 8
mpl.rcParams['xtick.major.size'] = 3
mpl.rcParams['ytick.major.size'] = 3
mpl.rcParams['xtick.major.width'] = .7
mpl.rcParams['ytick.major.width'] = .7
mpl.rcParams['xtick.direction'] = 'out'
mpl.rcParams['ytick.direction'] = 'out'
mpl.rcParams['figure.titlesize'] = 8
mpl.rc('text', usetex = True)

mpl.rcParams['text.latex.preamble'] = [
#    r'\renewcommand{\familydefault}{\sfdefault}',
#    r'\usepackage[scaled=1]{helvet}',
    r'\usepackage[helvet]{sfmath}',
#    r'\everymath={\sf}'
]

#mpl.rcParams['text.latex.preamble'] = [
#       r'\usepackage{helvet}',    # set the normal font here
#       r'\usepackage{sansmath}',  # load up the sansmath so that math -> helvet
#       r'\sansmath'               # <- tricky! -- gotta actually tell tex to use!
#]
#
##mpl.rcParams['font.family'] = 'sans-serif'
##mpl.rcParams['font.sans-serif'] = ["Helvetica"]
##mpl.rcParams['font.sans-serif'] = ['Arial', 'Tahoma', 'DejaVu Sans', 'Lucida Grande', 'Verdana']
#mpl.rcParams['font.family'] = 'sans-serif'
#mpl.rcParams['font.sans-serif'] = 'Helvetica'

fontsize = 8

#print('matplotlib: {}'.format(matplotlib.__version__))

def plotSpec(kVec, wVec, spec):
    spec = np.roll(spec, prms.chainLength // 2 - 1, axis=1)
    kVecPosNeg = np.linspace(np.pi, -np.pi, prms.chainLength, endpoint=False)
    kVec = np.flip(kVecPosNeg)

    fig = plt.figure()
    ax = fig.add_subplot(111)
    CS = ax.pcolormesh(kVec, -wVec, spec, cmap='pink', shading = 'gouraud', vmin=0.,vmax=0.1)
    cbar = fig.colorbar(CS, ax=ax)
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

    for axis in ['top', 'bottom', 'left', 'right']:
        ax.spines[axis].set_linewidth(0.5)
        axIn1.spines[axis].set_linewidth(0.5)
        axIn2.spines[axis].set_linewidth(0.5)
        axIn3.spines[axis].set_linewidth(0.5)


    lvls = np.logspace(np.log10(spec[-1, -1]) - 0.01, 0, 200)
    #CS = ax.contourf(kVecPosNeg, -wVec, spec, cmap='pink', norm=LogNorm(), levels=lvls)
    CS = ax.pcolormesh(kVecPosNeg, -wVec, spec, cmap='pink', norm=LogNorm(), shading = 'gouraud')
    #ax.plot(kVecPosNeg, 2. * prms.t * np.cos(kVecPosNeg), color = 'red')
    cbar = fig.colorbar(CS, ax=ax)
    cbar.set_ticks([])
    cbar.ax.minorticks_off()
    cbar.outline.set_linewidth(.5)
    #cbar.set_ticks([1e-0, 1e-1, 1e-2])
    #cbar.set_ticklabels([r'$10^{0}$', r'$10^{-1}$', r'$10^{-2}$'])

    kPoint = np.pi / 2.
    gsT = - 2. / np.pi * prms.chainLength
    z = 2. * eta ** 2 / prms.w0 * gsT
    wTilde = prms.w0 * np.sqrt(1. - z)
    wDash = prms.w0 * (1. - z)
    fac = .5 * (2. - z) / (1. - z)
    #selfE = - fac * eta ** 2 / wDash * (- 2. * prms.t * np.sin(kPoint) ** 2)
    selfE = eta**2 * prms.w0 / (wTilde**2) * (- 2. * prms.t * np.sin(kPoint))**2


    dynLoc = (1. - 0.5 * eta**2 * prms.w0 / wTilde)

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

    plt.savefig('spectralGSNum.png', format='png', bbox_inches='tight', dpi = 600)
    #plt.tight_layout()
    #plt.show()


def plotSpecLogDashed(wVec, spec, eta):
    spec = spec + 1e-12

    spec = np.roll(spec, prms.chainLength // 2 - 1, axis=1)
    spec = np.abs(spec)
    kVecPosNeg = np.linspace(np.pi, -np.pi, prms.chainLength, endpoint=False)
    kVecPosNeg = np.flip(kVecPosNeg)

    fig = plt.figure()
    fig.set_size_inches(4., 2.25)

    ax = fig.add_subplot(111)

    for axis in ['top', 'bottom', 'left', 'right']:
        ax.spines[axis].set_linewidth(0.0)

    terrainCmap = cm.get_cmap('terrain', 1500)
    newcolors = np.vstack((terrainCmap(np.linspace(0, 0.3, 128)),
                           terrainCmap(np.linspace(0.3, 0.9, 1024))
                           ))
    newcmp = matplotlib.colors.ListedColormap(newcolors)



    lvls = np.logspace(np.log10(spec[-1, -1]) - 0.01, 0, 200)
    #CS = ax.pcolormesh(kVecPosNeg, -wVec, spec, cmap='pink', norm=LogNorm(), shading = 'gouraud')
    CS = ax.pcolormesh(kVecPosNeg, -wVec, spec, cmap=newcmp, norm=LogNorm(), shading = 'gouraud')
    cbar = fig.colorbar(CS, ax=ax)
    #cbar.set_ticks([])
    #cbar.ax.minorticks_off()
    cbar.outline.set_linewidth(.5)

    kVec = np.linspace(np.pi, -np.pi, endpoint=False)
    kVec = np.flip(kVec)
    origBand = 2. * prms.t * np.cos(kVec)
    shakeP1 = 2. * prms.t * np.cos(kVec) + prms.w0
    shakeM1 = 2. * prms.t * np.cos(kVec) - prms.w0
    ax.plot(kVec, origBand, color = 'white', linestyle = '--', linewidth = 1.)
    ax.plot(kVec, shakeP1, color = 'white', linestyle = '--', linewidth = 1.)
    ax.plot(kVec, shakeM1, color = 'white', linestyle = '--', linewidth = 1.)

    ax.set_ylim(-4.1, 4.1)

    ax.set_xticks([-np.pi / 2., 0., np.pi / 2., np.pi])
    ax.set_xticklabels([r'$-\frac{\pi}{2}$', r'0', r'$\frac{\pi}{2}$', r'$\pi$'])
    ax.tick_params(axis='both', which='major', labelsize=fontsize)

    ax.set_xlabel('$k$', fontsize=fontsize)
    ax.set_ylabel('$\omega \, [t_h]$', fontsize=fontsize, labelpad = -1)

    ax.vlines(3. / 8. * np.pi, -4.1, 4.1, color = 'black', linestyle = '--', linewidth = 1.)
    plt.gcf().text(0.52, 0.9, r'$k = \frac{3 \pi}{8}$', fontsize=fontsize)


    yLimBot = spec[-1, prms.chainLength //2] - 0.0001
    yLimTop = spec[-1, prms.chainLength //2] + .4 # + 0.01 for second shakeoff

    cbar.ax.set_ylabel(r'$A(k, \omega)$', rotation=270, fontsize=fontsize, labelpad=10)
    #cbar.ax.tick_params(labelsize=fontsize)

    plt.savefig('Fig3a.png', format='png', bbox_inches='tight', dpi = 600)
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


#def plotPtGSWithCoh(ptGSL, ptGSS, eta, T):
#    #cohState = coherentState.getCoherentStateForN(N)
#    ptGSL = ptGSL
#    cohState = coherentState.getSqueezedState(eta, T) + 1e-32
#    fig = plt.figure()
#    fig.set_size_inches(3., 2.)
#
#    #ptGSL[1::2] = 1e-16
#    #ptGSS[1::2] = 1e-16
#
#    ax1 = plt.subplot2grid((20, 1), (0, 0), fig = fig, rowspan = 16)
#    ax2 = plt.subplot2grid((20, 1), (16, 0), fig = fig, rowspan = 1)
#    ax3 = plt.subplot2grid((20, 1), (17, 0), fig = fig, rowspan = 3)
#    #ax1 = fig.add_subplot(311)
#    #ax2 = fig.add_subplot(312)
#    #ax3 = fig.add_subplot(313)
#
#
#    for axis in ['top', 'bottom', 'left', 'right']:
#        ax1.spines[axis].set_linewidth(0.5)
#        ax2.spines[axis].set_linewidth(0.0)
#        ax3.spines[axis].set_linewidth(0.5)
#
#    ax1.spines['bottom'].set_linewidth(0.)
#    ax3.spines['top'].set_linewidth(0.)
#
#    ax1.spines[axis].set_linewidth(0.5)
#
#    myblue1 = '#3A5980'
#    myblue2 = '#3A6380'
#    myblue3 = '#3A6980'
#    myblue4 = '#335A80'
#    myblue5 = '#406080'
#
#    myyellow1 = '#CDB85A'
#    myyellow2 = '#E6CF65'
#    myyellow3 = '#E6C665'
#    myyellow4 = '#E6BB65'
#    myyellow5 = '#E6CC7F'
#
#    cmap = plt.cm.get_cmap('pink')
#
#    frontBarColor = myblue5
#    backBarColor = myyellow4
#    crossColor = 'black'
#
#
#    bins = np.arange(len(ptGSL))
#    ax1.bar(bins, np.abs(ptGSS), log=True, fc = backBarColor, label ="$L = 110$", width = 0.8, edgecolor ='black', linewidth = 0.25)
#    ax1.bar(bins, np.abs(ptGSL), log=True, fc=frontBarColor, label ="$L = 1010$", width = 0.8, edgecolor ='black', linewidth = 0.25)
#    #ax1.plot(bins[:42], np.abs(cohState)[:42], linestyle = '', marker = 'x', label = "Squeezed state", markersize = 3., markeredgecolor = crosscolor, markeredgewidth = .9)
#    ax1.plot(bins[:42], np.abs(cohState)[:42], linestyle = '', marker = 'x', color = crossColor, label = "Squeezed state", markersize = 4., markeredgecolor = 'black', markeredgewidth = 1.)
#    #ax.hlines(1., -2., 30, linestyles='--', colors='gray')
#
#    ax3.bar(bins, np.abs(ptGSS), log=True, fc = backBarColor, label ="$L = 110$", width = 0.8, edgecolor ='black', linewidth = 0.25)
#    ax3.bar(bins, np.abs(ptGSL), log=True, fc=frontBarColor, label ="$L = 1010$", width = 0.8, edgecolor ='black', linewidth = 0.25)
#    ax3.plot(bins[1:42:2], np.abs(cohState)[1:42:2], linestyle = '', marker = 'x', color = crossColor, label = "Squeezed state", markersize = 4., markeredgewidth = 1., clip_on = False, zorder = 100)
#
#
#    #ax3.hlines(1e-15, -1, 45, color = 'black', linestyle = '--', linewidth = .5)
#
#
#    ax1.set_ylabel(r'$P(n_{\mathrm{phot}})$', fontsize = fontsize)
#    ax3.set_xlabel(r'$n_{\mathrm{phot}}$', fontsize = fontsize)
#
#    ax1.set_xlim(-1, 45)
#    ax2.set_xlim(-1, 45)
#    ax3.set_xlim(-1, 45)
#    ax1.set_xticks([])
#
#    ax2.set_xticks([])
#
#    ax3.set_xticks([0, 10, 20, 30, 40])
#    ax3.set_xticklabels(['$0$', '$10$', '$20$', '$30$', '$40$'], fontsize = fontsize)
#
#
#    ax1.set_ylim(.5 * 1e-10, 1e1)
#    ax2.set_ylim(0., 1e-12)
#    ax3.set_ylim(1e-32, 1e-12)
#
#    ax1.set_yticks([1e0, 1e-3, 1e-6, 1e-9])
#    ax1.set_yticklabels(['$10^{0}$', '$10^{-3}$', '$10^{-6}$', '$10^{-9}$'])
#
#
#    ax3.set_yticks([1e-15, 1e-30])
#    ax3.set_yticklabels(['$10^{-15}$', '$10^{-30}$'])
#
#
#    ax2.set_yticks([])
#
#    legend = ax1.legend(fontsize = fontsize - 2, loc = 'upper left', bbox_to_anchor=(0.3, 1.), edgecolor = 'black', ncol = 1)
#    legend.get_frame().set_alpha(1.0)
#    legend.get_frame().set_boxstyle('Square', pad=0.0)
#    legend.get_frame().set_linewidth(0.5)
#
#
#    #plt.show()
#    plt.savefig('Fig1c.png', format='png', bbox_inches='tight', dpi = 600)

def plotPtGSWithCoh(ptGSL, ptGSS, eta, T):
    #cohState = coherentState.getCoherentStateForN(N)
    cohState = coherentState.getSqueezedState(eta, T) + 1e-32
    fig = plt.figure()
    fig.set_size_inches(3., 2.)

    ax = fig.add_subplot(111)

    left, bottom, width, height = [0.75, 0.4, 0.4, 0.55]
    axIn1 = fig.add_axes([left, bottom, width, height])

#    ax1 = plt.subplot2grid((1, 36), (0, 0), fig = fig, colspan = 15)
#    ax2 = plt.subplot2grid((1, 36), (0, 21), fig = fig, colspan = 15)


    for axis in ['top', 'bottom', 'left', 'right']:
        ax.spines[axis].set_linewidth(0.5)
        axIn1.spines[axis].set_linewidth(0.5)

    for axis in ['top', 'right']:
        ax.spines[axis].set_linewidth(0.0)
        axIn1.spines[axis].set_linewidth(0.0)


    #ax1p5.set_yticks([])
    #ax1p5.set_xticks([])

    myblue5 = '#406080'

    myyellow4 = '#E6BB65'

    cmap = plt.cm.get_cmap('pink')

    frontBarColor = myblue5
    backBarColor = myyellow4
    crossColor = 'black'


    bins = np.arange(len(ptGSL))
    ax.bar(bins - 0.25, np.abs(ptGSS)**2, log=True, fc = backBarColor, label ="$L = 10$", width = .5, edgecolor ='black', linewidth = 0.25)
    ax.bar(bins + 0.25, np.abs(ptGSL)**2, log=True, fc=frontBarColor, label ="$L = 510$", width = 0.5, edgecolor ='black', linewidth = 0.25)
    #ax.bar(bins, np.abs(ptGSL)**2, log=True, fc=frontBarColor, label ="$L = 510$", width = 0.8, edgecolor ='black', linewidth = 0.25)
    #ax1.plot(bins[:42], np.abs(cohState)[:42], linestyle = '', marker = 'x', label = "Squeezed state", markersize = 3., markeredgecolor = crosscolor, markeredgewidth = .9)
    ax.plot(bins[:26], np.abs(cohState)[:26]**2, linestyle = '', marker = 'x', color = crossColor, label = "Squeezed state", markersize = 5., markeredgecolor = crossColor, markeredgewidth = 1.)
    #ax.hlines(1., -2., 30, linestyles='--', colors='gray')

    axIn1.bar(bins - 0.25, np.abs(ptGSS)**2, log=False, fc = backBarColor, label ="$L = 10$", width = .5, edgecolor ='black', linewidth = 0.25)
    axIn1.bar(bins + 0.25, np.abs(ptGSL)**2, log=False, fc=frontBarColor, label ="$L = 510$", width = 0.5, edgecolor ='black', linewidth = 0.25)
    #axIn1.bar(bins, np.abs(ptGSL)**2, log=False, fc=frontBarColor, label ="$L = 510$", width = 0.8, edgecolor ='black', linewidth = 0.25)
    #ax2.plot(bins[:11], np.abs(cohState)[:11], linestyle = '', marker = 'x', color = crossColor, label = "Squeezed state", markersize = 3., markeredgewidth = 1., clip_on = False, zorder = 100)
    axIn1.plot(bins[:8], np.abs(cohState)[:8]**2, linestyle = '', marker = 'x', color = crossColor, label = "Squeezed state", markersize = 5., markeredgewidth = 1., clip_on = False, zorder = 100)


    #ax3.hlines(1e-15, -1, 45, color = 'black', linestyle = '--', linewidth = .5)


    ax.set_ylabel(r'$P(n_{\mathrm{phot}})$', fontsize = fontsize)
    ax.set_xlabel(r'$n_{\mathrm{phot}}$', fontsize = fontsize)

    axIn1.set_ylabel(r'$P(n_{\mathrm{phot}})$', fontsize = fontsize)
    axIn1.set_xlabel(r'$n_{\mathrm{phot}}$', fontsize = fontsize)


    ax.set_xlim(-1, 26)
    axIn1.set_xlim(-1, 7)
    #axIn1.set_xlim(-1, 43)
    ax.set_xticks([])

    axIn1.set_xticks([0, 2, 4, 6])

    ax.set_xticks([0, 10, 20])
    #ax.set_xticklabels(['$0$', '$10$', '$20$', '$30$', '$40$'], fontsize = fontsize)
    ax.set_xticklabels(['0', '10', '20'], fontsize = fontsize)


    ax.set_ylim(.5 * 1e-10, 2. * 1e0)

    ax.set_yticks([1e0, 1e-3, 1e-6, 1e-9])
    ax.set_yticklabels(['$10^{0}$', '$10^{-3}$', '$10^{-6}$', '$10^{-9}$'])
    #ax.set_yticklabels(['10^0', '10^-3', '10-6', '10^-9'])


    legend = ax.legend(fontsize = fontsize - 2, loc = 'upper left', bbox_to_anchor=(0.15, 1.1), edgecolor = 'black', ncol = 1)
    legend.get_frame().set_alpha(1.0)
    legend.get_frame().set_boxstyle('Square', pad=0.0)
    legend.get_frame().set_linewidth(0.5)

    #axis arrow
    arrow = patches.FancyArrowPatch((24, .5 * 1e-10), (27, .5*1e-10), arrowstyle='->', mutation_scale=7, zorder = 100, linewidth=.5, color = 'black', clip_on = False)
    ax.add_patch(arrow)

    arrow = patches.FancyArrowPatch((-1, 1e0), (-1, .8 * 1e1), arrowstyle='->', mutation_scale=7, zorder = 100, linewidth=.5, color = 'black', clip_on = False)
    ax.add_patch(arrow)

    #axis arrow
    arrow = patches.FancyArrowPatch((6, 0), (8, 0), arrowstyle='->', mutation_scale=7, zorder = 100, linewidth=.5, color = 'black', clip_on = False)
    axIn1.add_patch(arrow)

    arrow = patches.FancyArrowPatch((-1, 0.8), (-1, 1.), arrowstyle='->', mutation_scale=7, zorder = 100, linewidth=.5, color = 'black', clip_on = False)
    axIn1.add_patch(arrow)

    #tickPatch = patches.Rectangle((2, 1e-2), width = 0., height = 1e-2, linewidth = 0.7, clip_on = False)
    #ax.add_patch(tickPatch)

    #plt.show()
    plt.savefig('Fig1c.png', format='png', bbox_inches='tight', dpi = 600)




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

    #wVec = np.linspace(-4., 4., 2000, endpoint=False)
    #kVec = np.linspace(-np.pi, np.pi, 17, endpoint=True)


    gfNonEq = np.abs(gfNonEq) + 1e-16
    gfFloquet = np.abs(gfFloquet) + 1e-16
    fig, ax = plt.subplots(nrows=1, ncols=2, gridspec_kw={'width_ratios': [1, 20]})
    fig.set_size_inches(16./2., 9./2.)
    ax[0].tick_params(left=False, labelleft=False, bottom=False, labelbottom = False)
    ax[0].axis('off')
    ax = ax[1]

    #left, bottom, width, height = [0.125, 0.625, 0.3, 0.3]
    #axIn1 = fig.add_axes([left, bottom, width, height])
#
    #for axis in ['top', 'bottom', 'left', 'right']:
    #    axIn1.spines[axis].set_linewidth(1.5)
#
    #left, bottom, width, height = [0.74, 0.625, 0.3, 0.3]
    #axIn2 = fig.add_axes([left, bottom, width, height])
#
    #for axis in ['top', 'bottom', 'left', 'right']:
    #    axIn2.spines[axis].set_linewidth(1.5)



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
                ax.plot(wVec, quantumPlot[lInd, kInd, :], marker='', color=color, linestyle='-', linewidth = 1., label=labelString)
#                axIn1.plot(wVec, quantumPlot[lInd, kInd, :], marker='', color=color, linestyle='-', linewidth = 3.)
#                axIn2.plot(wVec, quantumPlot[lInd, kInd, :], marker='', color=color, linestyle='-', linewidth = 3.)
            else:
                ax.plot(wVec, quantumPlot[lInd, kInd, :], marker='', color=color, linestyle='-', linewidth = 1.)
#                axIn1.plot(wVec, quantumPlot[lInd, kInd, :], marker='', color=color, linestyle='-', linewidth = 3.)
#                axIn2.plot(wVec, quantumPlot[lInd, kInd, :], marker='', color=color, linestyle='-', linewidth = 3.)

        #plot floquet
#        if (kInd == 0):
#            ax.plot(wVec, floquetPlot[kInd, :], marker='', color='black', linestyle='-', linewidth = 0.5, label="Floquet")
#            axIn1.plot(wVec, floquetPlot[kInd, :], marker='', color='black', linestyle='-', linewidth = 0.5)
#            axIn2.plot(wVec, floquetPlot[kInd, :], marker='', color='black', linestyle='-', linewidth = 0.5)
#        else:
#            ax.plot(wVec, floquetPlot[kInd, :], marker='', color='black', linestyle='-', linewidth = 0.5)
#            axIn1.plot(wVec, floquetPlot[kInd, :], marker='', color='black', linestyle='-', linewidth = 0.5)
#            axIn2.plot(wVec, floquetPlot[kInd, :], marker='', color='black', linestyle='-', linewidth = 0.5)

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
    #arrow = patches.FancyArrowPatch((-2.8, 35), (-0.8, 35), arrowstyle='<->', mutation_scale=20, zorder = 100, linewidth=2., color = 'black')
    #ax.add_patch(arrow)
    #ax.text(-1.95, 45, "$\Omega$", fontsize = fontsize + 4)

    #rect = patches.Rectangle((-2.6, 2.), 1.5, 12.5, linewidth=1.5, edgecolor='black', facecolor='none', zorder = 100)
    #ax.add_patch(rect)

#    axIn1.set_ylim(2., 2. + 12.5)
#    axIn1.set_yscale('log', subsy = [0])
#    axIn1.set_yticks([])
#    axIn1.set_xlim(-2.65, -2.65 + 1.5)
#    axIn1.set_xticks([])
#    axIn1.set_xticklabels([])
#    axIn1.tick_params(axis='x', which='major', labelsize=14, width = 1.)

    #rect = patches.Rectangle((.5, 1500), 1.5, 8000, linewidth=1.5, edgecolor='black', facecolor='none', zorder = 100)
    #ax.add_patch(rect)

#    axIn2.set_ylim(1500., 1500. + 8000)
#    axIn2.set_yscale('log', subsy = [0])
#    axIn2.set_yticks([])
#    axIn2.set_xlim(.5, .5 + 1.5)
#    axIn2.set_xticks([])
#    axIn2.set_xticklabels([])
#    axIn2.tick_params(axis='x', which='major', labelsize=14, width = 1.)


    #plt.savefig('waterfallWithInsets.pdf', format='pdf', bbox_inches='tight')
    plt.savefig('waterfallNew.png', format='png', bbox_inches='tight', dpi = 600)
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
    ax.set_ylabel('$\log(A(k, \omega))$', fontsize = fontsize + 4)
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



def quantumToFloquetCrossover(wVec, gfArr, gfFloq, etaArr, nArr):

    assert(prms.chainLength == 90)

    fig = plt.figure()
    fig.set_size_inches(.65 * 4., .65 * 4.)
    ax = fig.add_subplot(111)

    for axis in ['bottom', 'left']:
        ax.spines[axis].set_linewidth(0.5)

    for axis in ['top', 'right']:
        ax.spines[axis].set_linewidth(0.0)


    ###make color array
    colorArr = np.zeros((len(etaArr), 3))
    startColor = (0.1, 0.35, .9)
    endColor = (.85, 1., 0.0)
    for colorInd, color in enumerate(colorArr):
        linFac = colorInd / (len(colorArr) - 1)
        tempColor = np.array((startColor[0] + linFac * (endColor[0] - startColor[0]), startColor[1] + (endColor[1] - startColor[1]) * linFac, startColor[2] + (endColor[2] - startColor[2]) * linFac ))
        colorArr[colorInd, :] = tempColor

    colorArr = np.flip(colorArr, axis = 0)
    #print(colorArr)

    cmap = plt.cm.get_cmap('gist_earth')
    #colorArr = [cmap(0.05), cmap(0.15), cmap(0.25), cmap(0.4), cmap(0.6), cmap(0.8)]
    for etaInd, eta in enumerate(etaArr):
        #print(eta)
        etaLabel = eta * np.sqrt(prms.chainLength)
        #color = colorArr[etaInd]
        #color = cmap(etaInd / (len(etaArr) + 1))
        #color = 'black'
        hsv = (colorArr[etaInd][0], colorArr[etaInd][1], colorArr[etaInd][2])
        color = matplotlib.colors.hsv_to_rgb(hsv)
        labelStr = r'$g = {:.2f}$'.format(etaLabel) + r', $| \alpha |^2$' + ' $= {:.1f}$'.format(nArr[etaInd])
        ax.plot(-wVec, np.imag(gfArr[etaInd, :]) * (.35 * 1e1) ** etaInd, color = color, linewidth = 1.2)

        boxProps = dict(boxstyle='square', facecolor='white', alpha=1., linewidth = 0., fill = True, pad = 0.15)

        color = 'black'

        if(etaInd == 0):
            plt.gcf().text(.8, 0.135, "${:.0f}$".format(nArr[etaInd]), fontsize=8, color = color, alpha = 1., bbox = boxProps)

        if(etaInd == 3):
            plt.gcf().text(.8, 0.2275, "${:.2f}$".format(nArr[etaInd]), fontsize=8, color = color, alpha = 1., bbox = boxProps)

        if(etaInd == 6):
            plt.gcf().text(.8, 0.3175, "${:.2f}$".format(nArr[etaInd]), fontsize=8, color = color, alpha = 1., bbox = boxProps)

        if(etaInd == 9):
            plt.gcf().text(.8, 0.4075, "${:.1f}$".format(nArr[etaInd]), fontsize=8, color = color, alpha = 1., bbox = boxProps)

        if(etaInd == 12):
            plt.gcf().text(.8, 0.4975, "${:.1f}$".format(nArr[etaInd]), fontsize=8, color = color, alpha = 1., bbox = boxProps)

        if(etaInd == 15):
            plt.gcf().text(.8, 0.5875, "${:.1f}$".format(nArr[etaInd]), fontsize=8, color = color, alpha = 1., bbox = boxProps)

        if(etaInd == 19):
            plt.gcf().text(.8, 0.705, "${:.0f}$".format(nArr[etaInd]), fontsize=8, color = color, alpha = 1., bbox = boxProps)

    ax.plot(-wVec, np.imag(gfFloq) * (.35 * 1e1) ** (len(etaArr) - 1), color = 'red', label = 'Floquet', linewidth = 1.1, linestyle = '--', dashes = (3., 2.))


    plt.gcf().text(.14, 0.705, r'$\mathrm{Floquet}$', fontsize=8, color = 'red', alpha = 1., bbox = boxProps)
    plt.gcf().text(.725, .775, r"$\underline{\Delta N_{\mathrm{phot}}^{\mathrm{pump}}}$", fontsize=fontsize, color = 'black')


    peakPos = 2. * prms.t * np.cos(3. / 8. * np.pi)

    ax.set_yscale('log')
    ax.set_xlim(-4., 2.5)
    ax.set_ylabel(r"$A(k = \frac{3}{8}\pi, \omega)$", fontsize = fontsize, labelpad = 0)
    ax.set_xlabel(r"$\omega {-} \varepsilon(k) \, [t_h]$", fontsize = fontsize)
    yLimBot = 1e-2
    yLimTop = 1e12
    ax.set_ylim(yLimBot, yLimTop)
    epsK = 2. * prms.t * np.cos(3. / 8. * np.pi)

    ax.set_xticks([epsK - 3. * prms.w0, epsK - 2. * prms.w0, epsK - prms.w0, epsK, epsK + prms.w0, epsK + 2. * prms.w0, epsK + 3. * prms.w0])
    ax.set_xticklabels([r'$-3$', r'$-2$', r'$-1$', r'$0$', r'$1$', r'$2$', r'$3$'], fontsize = 8)

    ax.set_yticks([1e-2, 1e1, 1e4, 1e7, 1e10])
    ax.set_yticklabels([r'$10^{-2}$', r'$10^{1}$', r'$10^{4}$', r'$10^{7}$', r'$10^{10}$'], fontsize = 8)

    vlineColor = '#7397C0'
    ax.vlines(peakPos, yLimBot, yLimTop, color = vlineColor, linestyle = '-', linewidth = 1.25)
    ax.vlines(peakPos + prms.w0, yLimBot, yLimTop, color = vlineColor, linestyle = '-', linewidth = 1.25)
    ax.vlines(peakPos - prms.w0, yLimBot, yLimTop, color = vlineColor, linestyle = '-', linewidth = 1.25)

    #legend = ax.legend(fontsize = fontsize - 4, loc = 'upper right', bbox_to_anchor=(1.1, 1.01), edgecolor = 'black', ncol = 1)
    #legend.get_frame().set_alpha(1.)
    #legend.get_frame().set_boxstyle('Square', pad=0.1)
    #legend.get_frame().set_linewidth(0.5)

    #epsString = r'$\varepsilon(k)$'
    epsStringPW = r'${+} \omega_0$'
    epsStringMW = r'${-} \omega_0$'
    #plt.gcf().text(0.475, 0.9, epsString, fontsize=6)
    plt.gcf().text(0.585, 0.8925, epsStringPW, fontsize=fontsize)
    plt.gcf().text(0.35, 0.8925, epsStringMW, fontsize=fontsize)

    #ax.arrow(-3., 1e10, 0., 1e1)
    #ax.arrow(-2.5, 1e7, 0., 1e10, length_includes_head = False, color = 'darkseagreen', width = 0.025, head_width = 0.2, head_length = 0.5, zorder = 100, shape = 'full')

    #arrow = patches.Arrow(-2, 1000, 2, 0, zorder = 100, width = 2)
    #arrow = patches.FancyArrowPatch((-3.25, 1e1), (-2.4, 5. * 1e9), arrowstyle='->', mutation_scale=10, connectionstyle="arc3,rad=.175", zorder = 100, linewidth=1.5, color = 'black')
    arrow = patches.FancyArrowPatch((-3.05, 1e1), (-2.2, 5. * 1e10), arrowstyle='->', mutation_scale=10, connectionstyle="arc3,rad=.175", zorder = 100, linewidth=1.5, color = 'black')
    ax.add_patch(arrow)

    #ax.arrow(2., 1e-1, 0.2, 0., length_includes_head=True, color='black', width=0.0,
    #         head_width=0.03, head_length=0.05, clip_on = False)

    arrow = patches.FancyArrowPatch((2.5, 1.0 * 1e-2), (2.6, 1.0 * 1e-2), arrowstyle='->', mutation_scale=5, zorder = 100, linewidth=.5, color = 'black', clip_on = False)
    ax.add_patch(arrow)

    arrow = patches.FancyArrowPatch((-4, 1e10), (-4, 1.0 * 1e13), arrowstyle='->', mutation_scale=5, zorder = 100, linewidth=.5, color = 'black', clip_on = False)
    ax.add_patch(arrow)

    #plt.text(-3.9, 5. * 1e9, r'$\mathrm{increased}$' + '\n' + '$\mathrm{pump}$' + '\n' + r'$\mathrm{decreased}$' + r'$g$' , fontsize = 6)
    plt.text(-3.4, 2. * 1e10, r'$\mathrm{increased}$' + '\n' + '$\mathrm{pump}$' , fontsize = fontsize)

    #ax.plot(1, 0, ">k", transform=ax.get_yaxis_transform(), clip_on=False)
    #ax.plot(0, 1, "^k", transform=ax.get_xaxis_transform(), clip_on=False)
    #ax.plot(1, 0, ">k", clip_on=False)
    #ax.plot(0, 1, "^k", clip_on=False)


    plt.savefig('Fig3b.png', format='png', bbox_inches='tight', dpi = 600)
    #plt.tight_layout()
    #plt.show()

def quantumToFloquetCrossoverAFS(wVec, gfArr, gfFloq, etaArr, nArr):

    fig = plt.figure()
    fig.set_size_inches(.75 * 4., .75 * 2.)
    ax = fig.add_subplot(111)

    for axis in ['top', 'bottom', 'left', 'right']:
        ax.spines[axis].set_linewidth(0.5)

    cmap = plt.cm.get_cmap('gist_earth')
    colorArr = [cmap(0.05), cmap(0.15), cmap(0.25), cmap(0.4), cmap(0.6), cmap(0.8)]
    for etaInd, eta in enumerate(etaArr):
        #print(eta)
        etaLabel = eta * np.sqrt(prms.chainLength)
        #color = cmap(nArr[etaInd] / (nArr[-1] + 0.5))
        color = colorArr[etaInd]
        labelStr = r'$g = {:.2f}$'.format(etaLabel) + r', $| \alpha |^2$' + ' $= {:.1f}$'.format(nArr[etaInd])
        ax.plot(-wVec, np.imag(gfArr[etaInd, :]) * (.45 * 1e1) ** etaInd, color = color, label = labelStr, linewidth = 0.7)
    ax.plot(-wVec, np.imag(gfFloq) * (.45 * 1e1) ** 5, color = 'black', label = 'Floquet', linewidth = .35, linestyle = '--')


    ax.set_yscale('log')
    ax.set_xlim(-3., 4.5)
    ax.set_ylabel(r"$A(k = \frac{5}{8}\pi, \omega)$", fontsize = 10, labelpad = 0)
    ax.set_xlabel(r"$\omega$", fontsize = 10)

    #ax.set_xticks([0., np.pi / 2., np.pi, 1.5 * np.pi, 2. * np.pi])
    #ax.set_xticklabels(['0', r'$\frac{\pi}{2}$', '$\pi$', r'$\frac{3\pi}{2}$', '$2 \pi$'], fontsize = fontsize)

    plt.savefig('crossoverAFS.png', format='png', bbox_inches='tight', dpi = 600)
    #plt.tight_layout()
    #plt.show()


def plotAnalyticalConductivity(eta1, eta2, eta3):

    gsT = - 2. / np.pi * prms.chainLength

    fac = np.sqrt(1 - 2. * eta1 * eta1 / (prms.w0) * gsT)

    omegaVec = np.linspace(-500, 500, 150000 , endpoint=True)
    #omegaVec = np.linspace(-50, 50, 15000 , endpoint=True)
    #omegaVec = np.linspace(-5, 5, 1500 , endpoint=True)
    delta = 0.05

    etas = np.linspace(0., 2., 16) * 1. / np.sqrt(prms.chainLength)
    etasLabels = np.linspace(0., 2., 16)
    #gsKinetics = -coherentState.gsEffectiveKineticEnergyArray(etas)
    gsKinetics = - calcConductivity.gsEffectiveKineticEnergyArrayNum(etas)

    #cond1 = calcConductivity.calcConductivityNum(omegaVec, delta, eta1)
    #cond2 = calcConductivity.calcConductivityNum(omegaVec, delta, eta2)
    #cond3 = calcConductivity.calcConductivityNum(omegaVec, delta, eta3)

    #cond1 = calcConductivity.calcConductivityAna(omegaVec, delta, eta1)
    #cond2 = calcConductivity.calcConductivityAna(omegaVec, delta, eta2)
    #cond3 = calcConductivity.calcConductivityAna(omegaVec, delta, eta3)

    #intConductivities = integratedConductivityArr(omegaVec, delta, etas)


    #saveConductivities(cond1, cond2, cond3)
    #saveIntConductivities(intConductivities)
    cond1, cond2, cond3 = loadConductivities()
    intConductivities = loadIntConductivities()

    cond1 = 2. * np.pi * cond1
    cond2 = 2. * np.pi * cond2
    cond3 = 2. * np.pi * cond3

    fig = plt.figure()
    fig.set_size_inches(0.7 * 4., 0.7 * 3.25)
    #fig.set_size_inches(4., 4.)

    ax = fig.add_subplot(111)

    #left, bottom, width, height = [0.675, 0.41, 0.4, 0.3]
    #axIn1 = fig.add_axes([left, bottom, width, height])

    left, bottom, width, height = [0.71, 0.45, 0.4, 0.4]
    axIn2 = fig.add_axes([left, bottom, width, height])

    myblue = '#406080'
    myyellow = '#E6BB65'
    myyellow = '#CD8F14'


    for axis in ['top', 'bottom', 'left', 'right']:
        ax.spines[axis].set_linewidth(0.5)
        axIn2.spines[axis].set_linewidth(0.5)

    ax.plot(omegaVec, np.real(cond3), color = 'black', linewidth = 1.5, linestyle = '-', label = "g = {}".format(0))
    #ax.plot(omegaVec, np.real(cond2), color = 'lightsteelblue', linewidth = .5, linestyle = '-', label = "g = {}".format(0.1))
    ax.plot(omegaVec, np.real(cond2), color = myyellow, linewidth = 1.5, linestyle = '--', dashes=(4.5, 1.2), label = "g = {}".format(0.3))
    ax.plot(omegaVec, np.real(cond1), color = myblue, linewidth =1.5, linestyle = '-', label = "g = {}".format(1))


    #ax.set_ylim(1e-6, 1e4)
    #ax.set_yscale('log')
    ax.set_xlim(- 1.2 * prms.w0 * fac , 1.2 * prms.w0 * fac)

    #ax.set_yticks([1e2, 1e0, 1e-2, 1e-4, 1e-6])
    ax.set_xticks([-2., -1., 0., 1., 2.])
    ax.set_xticklabels([r'$-2$', r'$-1$', r'$0$', r'$1$', r'$2$', ])
    ax.tick_params(axis='both', which='major', labelsize=fontsize)

    ax.set_xlabel('$\omega[t_h]$', fontsize=fontsize)
    ax.set_ylabel(r'Re$(\sigma)[\frac{e^2}{h}]$', fontsize=fontsize)

#    axIn1.plot(omegaVec, np.real(cond1), color = 'tan', linewidth = 1.5, linestyle = '-', label = "g = {}".format(1.))
#    axIn1.plot(omegaVec, np.real(cond2), color = 'lightsteelblue', linewidth = 1.5, linestyle = '-', label = "g = {}".format(0.1))
#    axIn1.plot(omegaVec, np.real(cond3), color = 'black', linewidth = .5, linestyle = '-', label = "g = {}".format(0))
#
#    axIn1.set_ylim(0.979, 1.001)
#    axIn1.set_xlim(-0.001, 0.001)
#
#    axIn1.set_yticks([0.98, 1.])
#    axIn1.set_xticks([])
#    #axIn1.set_xticklabels([r'$-2$', r'$-1$', r'$0$', r'$1$', r'$2$', ])
#    axIn1.tick_params(axis='both', which='major', labelsize=fontsize)

    #axIn2.plot(etasLabels, gsKinetics, color = 'black', linewidth = 1., label = r'\langle - e_{\mathrm{kin}} \rangle')
    axIn2.plot(etasLabels, gsKinetics, color = 'black', linewidth = 1.5, label = r'$\langle - e_{\mathrm{kin}} \rangle$')
    axIn2.plot(etasLabels, 1. / np.pi * intConductivities, color = 'red', linewidth = 1.5, linestyle = '--', label = r'$\int \sigma(\omega) d \omega$')
    yLimBot = np.amin(gsKinetics) - 0.001
    yLimTop = np.amax(gsKinetics) + 0.004
    axIn2.set_ylim(yLimBot, yLimTop)
    axIn2.set_yticks([0.633, 0.64])
    axIn2.set_yticklabels(['$0.633$', '$0.640$'], fontsize = fontsize)
    axIn2.set_xticks([0, 2])
    axIn2.set_xticklabels(['$g = 0$', '$g = 2$'], fontsize = fontsize)
    axIn2.vlines(0., yLimBot, yLimTop, color = 'black', linestyle = 'dotted', linewidth = 1.5)
    axIn2.vlines(0.3, yLimBot, yLimTop, color = myyellow, linestyle = 'dotted', linewidth = 1.5)
    axIn2.vlines(1., yLimBot, yLimTop, color = myblue, linestyle = 'dotted', linewidth = 1.5)
    axIn2.tick_params(axis='both', which='major', labelsize=fontsize)

    #axIn2.set_xlabel('$g$', fontsize=fontsize)
    axIn2.set_ylabel(r'$e[t_h]$', fontsize=fontsize, labelpad=-5)


#    D0 = np.amax(np.real(cond3))
#    print("D0 = {}".format(D0))
#    D1 = np.amax(np.real(cond1))
#    rec = patches.Rectangle((-0.25, D1), 0., D0 - D1, color = myblue, linewidth = 1.)
#    recTop = patches.Rectangle((-0.26, D0), 0.1, 0., color = myblue, linewidth = 1.)
#    recBot = patches.Rectangle((-0.26, D1), 0.1, 0., color = myblue, linewidth = 1.)
#    ax.add_patch(rec)
#    ax.add_patch(recTop)
#    ax.add_patch(recBot)
#    ax.text(-0.5, D1 + 10, "$\gamma(g {=} 1)D_0$", fontsize = fontsize, rotation = 90)

    legend = ax.legend(fontsize = fontsize, loc = 'upper center', bbox_to_anchor=(.19, 1.0), edgecolor = 'black', ncol = 1)
    legend.get_frame().set_alpha(0.0)
    legend.get_frame().set_boxstyle('Square', pad=0.0)
    legend.get_frame().set_linewidth(0)



    legend = axIn2.legend(fontsize = fontsize, loc = 'upper center', bbox_to_anchor=(.56, 1.04), edgecolor = 'black', ncol = 1)
    legend.get_frame().set_alpha(1.0)
    legend.get_frame().set_boxstyle('Square', pad=0.0)
    legend.get_frame().set_linewidth(.5)


    plt.savefig('conductivityGS.png', format='png', bbox_inches='tight', dpi = 600)
    #plt.tight_layout()
    #plt.show()


def plotAnalyticalConductivityImaginary(eta1, eta2, eta3):

    gsT = - 2. / np.pi * prms.chainLength

    fac = np.sqrt(1 - 2. * eta1 * eta1 / (prms.w0) * gsT)

    omegaVec = np.linspace(-500, 500, 150000 , endpoint=True)
    #omegaVec = np.linspace(-50, 50, 15000 , endpoint=True)
    delta = 0.02

    cond1, cond2, cond3 = loadConductivities()

    cond1 = 2. * np.pi * cond1
    cond2 = 2. * np.pi * cond2
    cond3 = 2. * np.pi * cond3

    etas = np.linspace(0., 2., 20) * 1. / np.sqrt(prms.chainLength)

    fig = plt.figure()
    fig.set_size_inches(0.7 * 4., 0.7 * 3.25)

    ax = fig.add_subplot(111)

    #left, bottom, width, height = [0.16, 0.65, 0.3, 0.2]
    #axIn1 = fig.add_axes([left, bottom, width, height])
#
    #left, bottom, width, height = [0.675, 0.14, 0.2, 0.2]
    #axIn2 = fig.add_axes([left, bottom, width, height])

    for axis in ['top', 'bottom', 'left', 'right']:
        ax.spines[axis].set_linewidth(0.5)
    #    axIn2.spines[axis].set_linewidth(0.5)

    myblue = '#406080'
    myyellow = '#E6BB65'
    myyellow = '#CD8F14'

    ax.plot(omegaVec, np.imag(cond3), color = 'black', linewidth = 1.5, linestyle = '-', label = "g = {}".format(0))
    ax.plot(omegaVec, np.imag(cond2), color = myyellow, linewidth = 1.5, linestyle = '--',dashes=(4.3, 1), label = "g = {}".format(0.1))
    ax.plot(omegaVec, np.imag(cond1), color = myblue, linewidth = 1.5, linestyle = '-', label = "g = {}".format(1.))

    #ax.set_ylim(1e-6, 1e4)
    #ax.set_yscale('log')
    ax.set_xlim(- 1.5 * prms.w0 * fac , 1.5 * prms.w0 * fac)

    #ax.set_yticks([1e2, 1e0, 1e-2, 1e-4, 1e-6])
    ax.set_xticks([-2., -1., 0., 1., 2.])
    ax.set_xticklabels([r'$-2$', r'$-1$', r'$0$', r'$1$', r'$2$', ])
    ax.tick_params(axis='both', which='major', labelsize=fontsize)

    ax.set_xlabel('$\omega[t_h]$', fontsize=fontsize)
    ax.set_ylabel(r'Im$(\sigma)[\frac{e^2}{h}]$', fontsize=fontsize)

    #axIn1.plot(omegaVec, np.imag(cond1), color = 'tan', linewidth = 1.5, linestyle = '-', label = "g = {}".format(1.))
    #axIn1.plot(omegaVec, np.imag(cond2), color = 'lightsteelblue', linewidth = 1.5, linestyle = '-', label = "g = {}".format(0.1))
    #axIn1.plot(omegaVec, np.imag(cond3), color = 'black', linewidth = .5, linestyle = '-', label = "g = {}".format(0))

    #xlimBot = -1.15
    #xlimTop = -0.85
    #ylimBot = -0.018
    #ylimTop = -0.002
    #axIn1.set_xlim(xlimBot, xlimTop)
    #axIn1.set_ylim(ylimBot, ylimTop)
    #axIn1.set_xticks([])
    #axIn1.set_yticks([])

    #rect = patches.Rectangle((xlimBot, -0.04), xlimTop - xlimBot, 0.07, linewidth=.4, edgecolor='black', facecolor='none', zorder = 100)
    #ax.add_patch(rect)
    #ax.plot([xlimBot, -2.055], [0.03, 0.425], linewidth = .5, color = 'black')
    #ax.plot([xlimTop, -0.3], [0.03, 0.425], linewidth = .5, color = 'black')

    #axIn2.plot(omegaVec, np.imag(cond1), color = 'tan', linewidth = 1.5, linestyle = '-', label = "g = {}".format(1.))
    #axIn2.plot(omegaVec, np.imag(cond2), color = 'lightsteelblue', linewidth = 1.5, linestyle = '-', label = "g = {}".format(0.1))
    #axIn2.plot(omegaVec, np.imag(cond3), color = 'black', linewidth = .5, linestyle = '-', label = "g = {}".format(0))

    #xlimBot = -0.007
    #xlimTop = -0.003
    #ylimBot = -1.001
    #ylimTop = -0.979
    #axIn2.set_xlim(xlimBot, xlimTop)
    #axIn2.set_ylim(ylimBot, ylimTop)
    #axIn2.set_xticks([])
    #axIn2.set_yticks([-1., -0.98])
    #axIn2.set_yticklabels(['${-}1.0$', '${-}0.98$'])
    #axIn2.tick_params(axis='both', which='major', labelsize=fontsize - 4, pad = 0)

    #rect = patches.Rectangle((xlimBot, ylimBot), xlimTop - xlimBot, ylimTop - ylimBot, linewidth=.4, edgecolor='black', facecolor='none', zorder = 100)
    #rect = patches.Rectangle((-0.06, -1.025), 0.1, 0.1, linewidth=.4, edgecolor='black', facecolor='none', zorder = 100)
    #ax.add_patch(rect)

    #legend = ax.legend(fontsize = fontsize, loc = 'upper center', bbox_to_anchor=(.25, 1.05), edgecolor = 'black', ncol = 1)
    #legend.get_frame().set_alpha(0.0)
    #legend.get_frame().set_boxstyle('Square', pad=0.1)
    #legend.get_frame().set_linewidth(0)

    plt.savefig('conductivityGSImaginary.png', format='png', bbox_inches='tight', dpi = 600)
    #plt.tight_layout()
    #plt.show()

def saveConductivities(cond1, cond2, cond3):
    file = h5py.File("data/conductivities.h5", 'w')
    file.create_dataset("cond1", data=cond1)
    file.create_dataset("cond2", data=cond2)
    file.create_dataset("cond3", data=cond3)
    file.close()


def saveIntConductivities(intConductivities):
    file = h5py.File("data/intConductivities.h5", 'w')
    file.create_dataset("intCond", data=intConductivities)
    file.close()

def loadConductivities():
    file = h5py.File("data/conductivities.h5", 'r')
    cond1 = file["cond1"][()]
    cond2 = file["cond2"][()]
    cond3 = file["cond3"][()]
    file.close()
    return (cond1, cond2, cond3)


def loadIntConductivities():
    file = h5py.File("data/intConductivities.h5", 'r')
    intCond = file["intCond"][()]
    file.close()
    return intCond

def integratedConductivityArr(omegaVec, delta, etas):
    intConductivity = np.zeros(len(etas))
    for indEta, eta in enumerate(etas):
        condTemp = np.real(calcConductivity.calcConductivityNum(omegaVec, delta, eta))
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
    fig.set_size_inches(0.6 * 4., 0.6 * 2.2)
    ax = fig.add_subplot(111)

    left, bottom, width, height = [0.875, 0.36, 0.4, 0.5]
    axIn1 = fig.add_axes([left, bottom, width, height])

    for axis in ['top', 'bottom', 'left', 'right']:
        ax.spines[axis].set_linewidth(0.5)
        axIn1.spines[axis].set_linewidth(0.5)

    cmap = plt.cm.get_cmap('viridis')

    colorArr = np.zeros((len(etas), 3))
    startColor = (0.075, 0.3, .9)
    endColor = (.75, .7, 0.3)
    for colorInd, color in enumerate(colorArr):
        linFac = colorInd / (len(colorArr) - 1)
        tempColor = np.array((startColor[0] + linFac * (endColor[0] - startColor[0]), startColor[1] + (endColor[1] - startColor[1]) * linFac, startColor[2] + (endColor[2] - startColor[2]) * linFac ))
        if(colorInd >= 1 and colorInd < 3):
            tempColor[0] += -0.05
        elif(colorInd > 3):
            tempColor[0] += +0.05
        if(colorInd == len(colorArr) - 1):
            tempColor[-1] = 0.
        colorArr[colorInd, :] = tempColor

    colorArr = np.flip(colorArr, axis = 0)

    xArr = np.linspace(0., 2. * np.pi, bins)
    for indEta in range(len(etas)):
        eta = etas[indEta]
        etaLabel = eta * np.sqrt(prms.chainLength)
        #color = cmap(etaLabel / (etas[-1] * np.sqrt(prms.chainLength) + 0.1))
        #color = colorArr[len(etas) - indEta - 1]
        hsv = (colorArr[indEta][0], colorArr[indEta][1], colorArr[indEta][2])
        color = matplotlib.colors.hsv_to_rgb(hsv)

        ax.plot(xArr, landscapes[indEta, :] / prms.chainLength, color = color, label = r'g = {:.2f}'.format(etaLabel), linewidth = 1.5)

        #if(etaLabel == 1.):
        #    eGSRand = 0.10872966341211296
        #    ax.plot(0., eGSRand, marker='X', color=color, markersize=8, clip_on = False, zorder = 100)

        axIn1.plot(Ls, photonOccsArb[:, indEta], color = color, linewidth = 1.)
        axIn1.plot(Ls, photonOccs2[:, indEta], color = 'black', linestyle = 'dotted', linewidth = 1.)

    #labelString = "$\omega$ = {:.2f}".format(prms.w0)
    #ax.text(0., .5, labelString, fontsize = 14)
    ax.set_ylim(-.7, 0.7)
    ax.set_ylabel("$e_{\psi_{\mathrm{T(FS)}}}[t_h]$", fontsize = fontsize, labelpad = -2)
    ax.set_xlabel("$\mathrm{FS}$ $\mathrm{center}$", fontsize = fontsize)

    ax.set_xlim(0., 2. * np.pi)
    ax.set_xticks([0., np.pi / 2., np.pi, 1.5 * np.pi, 2. * np.pi])
    ax.set_xticklabels(['0', r'$\frac{\pi}{2}$', '$\pi$', r'$\frac{3\pi}{2}$', '$2 \pi$'], fontsize = fontsize)

    yLimBot = -0.02
    yLimTop = 0.23
    axIn1.set_ylim(yLimBot, yLimTop)
    axIn1.vlines(1010, yLimBot, yLimTop, color = 'red', linestyle = '-', linewidth = .4)

    axIn1.set_xscale('log')
    axIn1.set_xlabel('$L$', fontsize = fontsize)
    axIn1.set_ylabel('$N_{\mathrm{phot}}$', fontsize = fontsize, labelpad = -2)

    axIn1.set_xticks([1e2, 1e4])
    axIn1.set_xticklabels(['$10^2$', '$10^4$'], fontsize = fontsize)
    axIn1.set_yticks([0.0, 0.2])
    axIn1.set_yticklabels(['$0.0$', '$0.2$'], fontsize = fontsize)

    legend = ax.legend(fontsize = fontsize - 2, loc = 'upper left', bbox_to_anchor=(-0.02, 1.35), edgecolor = 'black', ncol = 4)
    legend.get_frame().set_alpha(1.)
    legend.get_frame().set_boxstyle('Square', pad=0.1)
    legend.get_frame().set_linewidth(0.0)

    plt.savefig('fsShiftsAllOrders.png', format='png', bbox_inches='tight', dpi = 600)
    #plt.tight_layout()
    #plt.show()

def plotLandscapes1Order(etas, orderH):
    orderH = 1
    bins = 100

    #landscapes = fsShift.getManyEnergyLandscapes(etas, orderH, bins)
    #file = h5py.File("data/landscapesCrit" + str(orderH) + ".h5", 'w')
    #file.create_dataset("landscapes", data=landscapes)
    #file.create_dataset("etas", data=etas)
    #file.close()

    file = h5py.File("data/landscapesCrit" + str(orderH) + ".h5", 'r')
    landscapes = file["landscapes"][()]
    etas = file["etas"][()]
    file.close()

    Ls = np.logspace(1., 4., 11, endpoint = True)
    etasNonNorm = etas * np.sqrt(prms.chainLength)

    #photonOccs1 = fsShift.occupationsForLengths(Ls, etasNonNorm, 1, 500)
    #file = h5py.File("data/occsOneCrit.h5", 'w')
    #file.create_dataset("occs", data=photonOccs1)
    #file.close()

    exactShifts = np.array([0, 0, 0, np.arccos(np.pi / (4. * etas[3]**2 * 1010)), np.arccos(np.pi / (4. * etas[4]**2 * 1010)), np.arccos(np.pi / (4. * etas[5]**2 * 1010)), np.arccos(np.pi / (4. * etas[6]**2 * 1010))])
    shiftPointHeights = np.array([-0.05, -0.2, -0.35, -0.5, -0.73, -1.08, -1.55])

    file = h5py.File("data/occsOneCrit.h5", 'r')
    photonOccs1 = file["occs"][()]
    file.close()

    fig = plt.figure()
    #fig.set_size_inches(0.6 * 4., 0.6 * 3.)
    fig.set_size_inches(0.6 * 4., 0.6 * 2.5)


    ax = fig.add_subplot(111)

    left, bottom, width, height = [0.83, 0.7, 0.35, 0.4]
    axInTop = fig.add_axes([left, bottom, width, height])

    left, bottom, width, height = [0.83, 0.55, 0.35, 0.1]
    axInBot = fig.add_axes([left, bottom, width, height])

    left, bottom, width, height = [0.83, 0.65, 0.35, 0.05]
    axInMid = fig.add_axes([left, bottom, width, height])

    for axis in ['top', 'bottom', 'left', 'right']:
        ax.spines[axis].set_linewidth(0.5)
        axInTop.spines[axis].set_linewidth(0.5)
        axInBot.spines[axis].set_linewidth(0.5)


    colorArr = np.zeros((len(etas), 3))
    startColor = (0.075, 0.3, .9)
    endColor = (.75, .7, 0.3)
    for colorInd, color in enumerate(colorArr):
        linFac = colorInd / (len(colorArr) - 1)
        tempColor = np.array((startColor[0] + linFac * (endColor[0] - startColor[0]), startColor[1] + (endColor[1] - startColor[1]) * linFac, startColor[2] + (endColor[2] - startColor[2]) * linFac ))
        if(colorInd >= 1 and colorInd < 3):
            tempColor[0] += -0.05
        elif(colorInd > 3):
            tempColor[0] += +0.05
        if(colorInd == len(colorArr) - 1):
            tempColor[-1] = 0.
        colorArr[colorInd, :] = tempColor

    colorArr = np.flip(colorArr, axis = 0)


    xArr = np.linspace(0., 2. * np.pi, bins)
    for indEta in range(len(etas)):
        eta = etas[indEta]
        etaLabel = eta * np.sqrt(prms.chainLength)
        #color = cmap(etaLabel / (etas[-1] * np.sqrt(prms.chainLength) + 0.1))
        hsv = (colorArr[indEta][0], colorArr[indEta][1], colorArr[indEta][2])
        color = matplotlib.colors.hsv_to_rgb(hsv)

        ax.plot(xArr, landscapes[indEta, :] / prms.chainLength, color = color, label = r'$g = {:.2f}$'.format(etaLabel), linewidth = 1.5)
        ax.plot(exactShifts[indEta], shiftPointHeights[indEta], linestyle = '', marker = 'o', color = color, markersize = 4., markeredgecolor = 'black', markeredgewidth = 0.3, clip_on = False, zorder = 100)

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
    ax.set_ylim(-1.8, .9)
    ax.set_xlim(0., 2. * np.pi)
    ax.set_ylabel("$e_{\psi_{\mathrm{T(FS)}}}[t_h]$", fontsize = fontsize, labelpad = -2)
    ax.set_xlabel("$\mathrm{FS}$ $\mathrm{center}$", fontsize = fontsize)

    ax.set_xticks([0., np.pi / 2., np.pi, 1.5 * np.pi, 2. * np.pi])
    ax.set_xticklabels(['0', r'$\frac{\pi}{2}$', '$\pi$', r'$\frac{3\pi}{2}$', '$2 \pi$'], fontsize = fontsize)

    axInBot.set_xscale('log')
    axInMid.set_xscale('log')
    axInBot.set_xlabel('$L$', fontsize = fontsize)
    axInTop.set_ylabel('$N_{\mathrm{phot}}$', fontsize = fontsize, rotation = 0)
    axInTop.yaxis.set_label_coords(-0.24, .8)

    axInTop.set_xticks([])
    axInBot.set_yticks([0])
    axInMid.set_xticks([])
    axInMid.set_yticks(())
    #axInBot.set_yticklabels(['$N_{\mathrm{pt}} = 0$'], fontsize = 8)
    axInBot.set_yticklabels(['$0$'], fontsize = 8)

    axInBot.set_xticks([1e2, 1e4])
    axInBot.set_xticklabels(['$10^2$', '$10^4$'], fontsize = 8)

    axInTop.set_yticks([1e-2, 1e2])
    axInTop.set_yticklabels(['$10^{-2}$', '$10^2$'], fontsize = 8)

    yLimBot = 5. * 1e-4
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

    arrow = patches.FancyArrowPatch((0.7, 0.0), (0.1, -0.5), arrowstyle='->', mutation_scale=5, zorder = 101, linewidth=.5, color = 'black')
    ax.add_patch(arrow)
    ax.text(0.4, 0.05, "$g_c {+} \delta$", fontsize = fontsize)

    #arrow = patches.FancyArrowPatch((100, 1e-1), (400, 5. * 1e-2), arrowstyle='->', mutation_scale=5, zorder = 101, linewidth=.5, color = 'black')
    #axInTop.add_patch(arrow)
    #axInTop.text(15, 3. * 1e-2, "$g_c {+} \delta$", fontsize = fontsize - 1)
    plt.gcf().text(1.2, 0.84, "$g_c {+} \delta$", fontsize=8, color='black')

    #ax.spines['right'].set_visible(False)

    #legend = ax.legend(fontsize = fontsize, loc = 'upper left', bbox_to_anchor=(0., 1.12), edgecolor = 'black', ncol = 2)
    #legend.get_frame().set_alpha(0.95)
    #legend.get_frame().set_boxstyle('Square', pad=0.1)
    #legend.get_frame().set_linewidth(0.5)

    legend = axInTop.legend(fontsize = fontsize, loc = 'upper left', bbox_to_anchor=(-0.07, 1.12), edgecolor = 'black', ncol = 1)
    legend.get_frame().set_alpha(0.0)
    legend.get_frame().set_boxstyle('Square', pad=0.05)
    legend.get_frame().set_linewidth(0.0)

    plt.savefig('fsShifts1.png', format='png', bbox_inches='tight', dpi = 600)
    #plt.tight_layout()
    #plt.show()


def plotLandscapes2Order(etas, orderH):
    orderH = 2
    bins = 100

    landscapes = fsShift.getManyEnergyLandscapes(etas, orderH, bins)
    file = h5py.File("data/landscapesCrit" + str(orderH) + ".h5", 'w')
    file.create_dataset("landscapes", data=landscapes)
    file.create_dataset("etas", data=etas)
    file.close()

    #file = h5py.File("data/landscapes" + str(orderH) + ".h5", 'r')
    #landscapes = file["landscapes"][()]
    #etas = file["etas"][()]
    #file.close()

    fig = plt.figure()
    #fig.set_size_inches(0.6 * 16. / 4., 0.6 * 12 / 4.)
    fig.set_size_inches(0.6 * 4., 0.6 * 2.5)

    ax = fig.add_subplot(111)

    for axis in ['top', 'bottom', 'left', 'right']:
        ax.spines[axis].set_linewidth(0.5)

    colorArr = np.zeros((len(etas), 3))
    startColor = (0.075, 0.3, .9)
    endColor = (.75, .7, 0.3)
    for colorInd, color in enumerate(colorArr):
        linFac = colorInd / (len(colorArr) - 1)
        tempColor = np.array((startColor[0] + linFac * (endColor[0] - startColor[0]), startColor[1] + (endColor[1] - startColor[1]) * linFac, startColor[2] + (endColor[2] - startColor[2]) * linFac ))
        if(colorInd >= 1 and colorInd < 3):
            tempColor[0] += -0.05
        elif(colorInd > 3):
            tempColor[0] += +0.05
        if(colorInd == len(colorArr) - 1):
            tempColor[-1] = 0.
        colorArr[colorInd, :] = tempColor

    colorArr = np.flip(colorArr, axis = 0)

    xArr = np.linspace(0., 2. * np.pi, bins)


    print("arccos(- pi / 4) = {}".format(np.arccos(- np.pi / 4)))
    transitionPos1 = np.arccos(-np.pi / (4 * (etas[3:] * np.sqrt(1010))**2))
    transitionPos2 = 2. * np.pi - np.arccos(-np.pi / (4 * (etas[3:] * np.sqrt(1010))**2))

    ##find index for 'transition'
    indicesLow = np.zeros(len(transitionPos1))
    for transitionInd, transition in enumerate(transitionPos1):
        for xInd, x in enumerate(xArr):
            if (x < transition):
                continue
            else:
                indicesLow[transitionInd] = xInd
                break

    ##find index for 'transition'
    indicesHigh = np.zeros(len(transitionPos1))
    for transitionInd, transition in enumerate(transitionPos2):
        for xInd, x in enumerate(xArr):
            if (x < transition):
                continue
            else:
                indicesHigh[transitionInd] = xInd
                break



    #print(transitionPos1)
    #print(transitionPos2)

    print(indicesLow)
    print(indicesHigh)


    for indEta in range(len(etas)):
        eta = etas[indEta]
        etaLabel = eta * np.sqrt(prms.chainLength)
        #color = cmap(etaLabel / (etas[-1] * np.sqrt(prms.chainLength) + 0.1))
        hsv = (colorArr[indEta][0], colorArr[indEta][1], colorArr[indEta][2])
        color = matplotlib.colors.hsv_to_rgb(hsv)
        if(indEta == 0):
            ax.plot(xArr, landscapes[indEta, :] / prms.chainLength, color = color, label = r'$g = 0$', linewidth = 1.5)
        elif(indEta < 3):
            ax.plot(xArr, landscapes[indEta, :] / prms.chainLength, color = color, label = r'$g = {:.2f}$'.format(etaLabel), linewidth = 1.5)
        elif(indEta == 3):
            ax.plot(xArr[:int(indicesLow[indEta - 3])], landscapes[indEta, :int(indicesLow[indEta - 3])] / prms.chainLength, color = color, label = r'$g = g_c {+} \delta$', linewidth = 1.5)
            ax.plot(xArr[int(indicesHigh[indEta - 3]):], landscapes[indEta, int(indicesHigh[indEta - 3]):] / prms.chainLength, color = color, linewidth = 1.5)
            ax.plot(xArr[:], landscapes[indEta, :] / prms.chainLength, color = color, linewidth = 1.5, linestyle = 'dotted')

        elif(indEta == 6):
            ax.plot(xArr[:int(indicesLow[indEta - 3])], landscapes[indEta, :int(indicesLow[indEta - 3])] / prms.chainLength, color = color, label = r'$g = 2$', linewidth = 1.5)
            ax.plot(xArr[int(indicesHigh[indEta - 3]):], landscapes[indEta, int(indicesHigh[indEta - 3]):] / prms.chainLength, color = color, linewidth = 1.5)
            ax.plot(xArr[:], landscapes[indEta, :] / prms.chainLength, color = color, linewidth = 1.5, linestyle = 'dotted')

        else:
            ax.plot(xArr[:int(indicesLow[indEta - 3])], landscapes[indEta, :int(indicesLow[indEta - 3])] / prms.chainLength, color = color, label = r'$g = {:.2f}$'.format(etaLabel), linewidth = 1.5)
            ax.plot(xArr[int(indicesHigh[indEta - 3]):], landscapes[indEta, int(indicesHigh[indEta - 3]):] / prms.chainLength, color = color, linewidth = 1.5)
            ax.plot(xArr[:], landscapes[indEta, :] / prms.chainLength, color = color, linewidth = 1.5, linestyle = 'dotted')


    #labelString = "$\omega$ = {:.2f}".format(prms.w0)
    #ax.text(0., .5, labelString, fontsize = 14)
    ax.set_ylim(-2.5, 0.9)
    ax.set_xlim(0., 2. * np.pi)
    ax.set_ylabel("$e_{\psi_{\mathrm{T(FS)}}}[t_h]$", fontsize = fontsize)
    ax.set_xlabel("$\mathrm{FS}$ $\mathrm{center}$", fontsize = fontsize)

    ax.set_xticks([0., np.pi / 2., np.pi, 1.5 * np.pi, 2. * np.pi])
    ax.set_xticklabels(['0', r'$\frac{\pi}{2}$', '$\pi$', r'$\frac{3\pi}{2}$', '$2 \pi$'], fontsize = fontsize)

    #ax.spines['right'].set_visible(False)

    legend = ax.legend(fontsize = fontsize, loc = 'upper left', bbox_to_anchor=(1., 1.05), edgecolor = 'black', ncol = 1)
    legend.get_frame().set_alpha(0.)
    legend.get_frame().set_boxstyle('Square', pad=0.1)
    legend.get_frame().set_linewidth(0.0)

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

def arbitraryEDist():
    eta = 1.0 / np.sqrt(prms.chainLength)

    #np.random.seed(13)
    np.random.seed(17)

    LShort = 20

    gsEShort = np.zeros(LShort)
    gsEShort[:LShort//2] = 1.
    np.random.shuffle(gsEShort)

    LLong = int(1e6)
    gsLong = np.zeros(LLong)
    for indGS, gsOcc in enumerate(gsEShort):
        binLength = LLong // LShort
        if(gsOcc == 1):
            gsLong[indGS * binLength : (indGS + 1) * binLength] = 1.

    kVec = np.linspace(- np.pi, np.pi, LLong)



    LMed = int(1e3)
    gsMed = np.zeros(LMed)
    for indGS, gsOcc in enumerate(gsEShort):
        binLength = LMed // LShort
        if(gsOcc == 1):
            gsMed[indGS * binLength : (indGS + 1) * binLength] = 1.

    gsE = np.append(gsMed, np.array([0, 0, 1, 1, 1, 1, 1, 0, 0, 0]))
    print("gs Integral = {}".format(np.sum(gsE)))

    gsEnergy = photonState.energyFromState(gsE, eta, 3)

    print("gs e = {}".format(gsEnergy / prms.chainLength))


    fig = plt.figure()
    fig.set_size_inches(1., 0.75)
    ax = fig.add_subplot(111)

    for axis in ['top', 'bottom', 'left', 'right']:
        ax.spines[axis].set_linewidth(0.5)

    plt.ylim([-0.05, 1.5])
    plt.xlim(-np.pi, np.pi)

    ticks = (-np.pi, -np.pi/2, 0, np.pi/2, np.pi)
    ax.set_xticks(ticks, minor=False)
    ax.set_yticks((0, 1), minor=False)
    ax.set_xticklabels((r"$-\pi$", r"$-\frac{\pi}{2}$", r"$0$", r"$\frac{\pi}{2}$", r"$\pi$"), fontsize=fontsize - 2)
    ax.set_yticklabels((r"$0$", r"$1$"), fontsize=fontsize - 2)
    ax.set_xlabel(r"$k$", fontsize=fontsize - 2)
    ax.set_ylabel(r"$n(k)$", fontsize=fontsize - 2)

    bins = np.arange(len(gsEShort))
    #ax.bar(bins, gsE, log=False, color='black', width=1., fill=False, linewidth = 1.)
    ax.plot(kVec, gsLong, color = 'black')

    cmap = plt.cm.get_cmap('gist_earth')
    color = cmap(1. / 2.1)


    ax.plot(2.3, 1.25, marker = 'X', color = color, markersize = 8)
    ax.text(-2.8, 1.2, "$e = {:.2f}$".format(gsEnergy / prms.chainLength), fontsize=fontsize)

    plt.gcf().text(0.12, .91, r'$\mathrm{Rand. Distribution}$', fontsize=6)

    plt.savefig('arbDist.png', format='png', bbox_inches='tight', dpi = 600)
    #plt.show()

def plotShiftInsetes():


    #Square 1
    delta = 0.04
    deltav = 0.0415
    ticks = (-np.pi, -np.pi/2, 0, np.pi/2, np.pi)


    fig, ax = plt.subplots(dpi = 800)
    fig.set_size_inches(1., 0.75)

    for axis in ['top', 'bottom', 'left', 'right']:
        ax.spines[axis].set_linewidth(0.5)

    ax.set_xticks(ticks, minor=False)
    ax.set_yticks((0,1), minor=False)
    ax.set_xticklabels((r"$-\pi$", r"$-\frac{\pi}{2}$", r"$0$", r"$\frac{\pi}{2}$", r"$\pi$"), fontsize=fontsize)
    ax.set_yticklabels((r"$0$", r"$1$"), fontsize= fontsize)
    ax.set_xlabel(r"$k$", fontsize = fontsize)
    ax.set_ylabel(r"$n(k)$", fontsize=fontsize)
    plt.vlines(0, 0, 1.1, color = "grey", linestyle="--", linewidth=0.8)
    #arrow = patches.FancyArrowPatch((.1, 1.15), (0.3, 1.15), arrowstyle='->', mutation_scale=10, zorder = 100, linewidth=1.5, color = 'black')
    #ax.add_patch(arrow)
    plt.vlines(-np.pi/2, 0 - deltav, 1+deltav, color = "black", linewidth = 1.5)
    plt.vlines(np.pi/2, 0 - deltav, 1+deltav, color = "black", linewidth = 1.5)
    plt.hlines(0, -np.pi, -np.pi/2+delta, color = "black", linewidth = 1.5)
    plt.hlines(1, -np.pi/2, np.pi/2, color = "black", linewidth = 1.5)
    plt.hlines(0, np.pi/2-delta, np.pi, color = "black", linewidth = 1.5)
    plt.ylim([-0.05, 1.7])
    plt.xlim(-np.pi, np.pi)
    plt.text(-2.8, 1.35, "$\mathrm{FS}$ $\mathrm{center} = 0$", fontsize=fontsize)

    plt.gcf().text(0.12, .91, r'$n = \frac{1}{2}$', fontsize=fontsize)


    #plt.show()
    plt.savefig('shiftInset1.png', format='png', bbox_inches='tight', dpi = 600)


    #Square 2
    fig, ax = plt.subplots(dpi = 800)
    fig.set_size_inches(1., 0.75)

    for axis in ['top', 'bottom', 'left', 'right']:
        ax.spines[axis].set_linewidth(0.5)

    ax.set_xticks(ticks, minor=False)
    ax.set_yticks((0,1), minor=False)
    ax.set_xticklabels((r"$-\pi$", r"$-\frac{\pi}{2}$", r"$0$", r"$\frac{\pi}{2}$", r"$\pi$"), fontsize= fontsize)
    ax.set_yticklabels((r"$0$", r"$1$"), fontsize= fontsize)
    ax.set_xlabel(r"$k$", fontsize =  fontsize)
    ax.set_ylabel(r"$n(k)$", fontsize= fontsize)
    plt.vlines(np.pi/2, 0, 1.1, color = "grey", linestyle="--", linewidth=0.8)
    arrow = patches.FancyArrowPatch((0, 1.15), (np.pi/2, 1.15), arrowstyle='->', mutation_scale=10, zorder = 100, linewidth=1.5, color = 'black')
    ax.add_patch(arrow)
    plt.vlines(0, 0-deltav, 1+deltav, color = "black")
    plt.vlines(np.pi, 0 - deltav, 1+deltav, color = "black")
    plt.hlines(0, -np.pi, delta, color = "black")
    plt.hlines(1, 0, np.pi, color = "black")
    plt.ylim([-0.05, 1.7])
    plt.xlim(-np.pi, np.pi)
    plt.text(-2.9, 1.27, r"$\mathrm{FS}$ $\mathrm{center} = \frac{\pi}{2}$", fontsize=fontsize)

    #plt.show()
    plt.savefig('shiftInset2.png', format='png', bbox_inches='tight', dpi = 600)


    #Square 3
    fig, ax = plt.subplots(dpi = 800)
    fig.set_size_inches(1., 0.75)

    for axis in ['top', 'bottom', 'left', 'right']:
        ax.spines[axis].set_linewidth(0.5)

    ax.set_xticks(ticks, minor=False)
    ax.set_yticks((0,1), minor=False)
    ax.set_xticklabels((r"$-\pi$", r"$-\frac{\pi}{2}$", r"$0$", r"$\frac{\pi}{2}$", r"$\pi$"), fontsize= fontsize)
    ax.set_yticklabels((r"$0$", r"$1$"), fontsize= fontsize)
    ax.set_xlabel(r"$k$", fontsize = fontsize)
    ax.set_ylabel(r"$n(k)$", fontsize= fontsize)
    plt.vlines(np.pi - 0.1, 0, 1.1, color = "grey", linestyle="--", linewidth=0.8)
    arrow = patches.FancyArrowPatch((0, 1.2), (np.pi, 1.2), arrowstyle='->', mutation_scale=10, zorder = 100, linewidth=1.5, color = 'black')
    ax.add_patch(arrow)
    plt.vlines(-np.pi/2, 0 - deltav, 1+deltav, color = "black")
    plt.vlines(np.pi/2, 0 - deltav, 1+deltav, color = "black")
    plt.hlines(1, -np.pi, -np.pi/2, color = "black")
    plt.hlines(1, np.pi/2, np.pi, color = "black")
    plt.hlines(0, -np.pi/2-delta, np.pi/2+delta, color = "black")
    plt.ylim([-0.05, 1.7])
    plt.xlim(-np.pi, np.pi)
    plt.text(-2.85, 1.4, "$\mathrm{FS}$ $\mathrm{center} = \pi$", fontsize=fontsize)

    plt.savefig('shiftInset3.png', format='png', bbox_inches='tight', dpi = 600)
    #plt.show()


def finiteSizeErrors(x, e1, e2, e3, e4):
    fig, ax = plt.subplots(nrows=1, ncols=1)

    fig.set_size_inches(2., 1.75)

    for axis in ['top', 'bottom', 'left', 'right']:
        ax.spines[axis].set_linewidth(0.5)

    myblue = '#406080'
    myyellow = '#E6BB65'

    myred = '#C37161'

    ax.loglog(x, 1. / (x), marker = '', color = 'black', linestyle = '--', label = r'$1 / L$', linewidth = 1.)
    ax.loglog(x, 1. / (x**2), marker = '', color = 'black', linestyle = '--', label = r'$1 / L^2$', linewidth = 1.)
    ax.loglog(x, e3, marker = 'v', color = 'red', linestyle = '-', label = 'Mean error', linewidth = .5, markersize = 3., markeredgecolor = 'black', markeredgewidth = 0.25)
    ax.loglog(x, e4, marker = '<', color = 'mediumseagreen', linestyle = '-', label = 'Max error', linewidth = .5, markersize = 3., markeredgecolor = 'black', markeredgewidth = 0.25)
    ax.loglog(x, 2. * e2, marker = 'o', color = myblue, linestyle = '-', label = 'Max Error', linewidth = .5, markersize = 3., markeredgecolor = 'black', markeredgewidth = 0.25, markerfacecolor = myblue)
    ax.loglog(x, 2. * e1, marker = 's', color = myyellow, linestyle = '-', label = 'Mean Error', linewidth = .5, markersize = 3., markeredgecolor = 'black', markeredgewidth = 0.25, markerfacecolor = myyellow)
    #ax.loglog(x, 1. / (x * np.sqrt(x)) * 2. * 1e1 , marker = '', color = 'black', linestyle = '--', label = r'$\sim 1 / L^{\frac{3}{2}}$')
    #ax.loglog(x, 1. / (x) , marker = '', color = 'gray', linestyle = '--', label = r'$\sim 1 / L$')


    plt.gcf().text(.925, 0.7, r"$\mathrm{Max}$", fontsize=fontsize - 2, color=myblue, alpha=1.)#, bbox=boxProps)
    plt.gcf().text(.925, 0.6, r"$\mathrm{Mean}$", fontsize= fontsize - 2, color=myyellow, alpha=1.)#, bbox=boxProps)
    plt.gcf().text(.925, 0.5, r"$1 / L$", fontsize=fontsize - 2, color='black', alpha=1.)#, bbox=boxProps)

    plt.gcf().text(.925, 0.34, r"$\mathrm{Max}$", fontsize=fontsize - 2, color=myblue, alpha=1.)#, bbox=boxProps)
    plt.gcf().text(.925, 0.225, r"$\mathrm{Mean}$", fontsize= fontsize - 2, color=myyellow, alpha=1.)#, bbox=boxProps)
    plt.gcf().text(.925, 0.125, r"$1 / L^2$", fontsize=fontsize - 2, color='black', alpha=1.)#, bbox=boxProps)

    ax.set_xticks([1e3, 1e5])
    ax.set_xticklabels(['$10^{3}$', '$10^{5}$'], fontsize = fontsize - 2)

    ax.set_ylim(1e-13, 1e0)
    ax.set_yticks([1e0, 1e-4, 1e-8, 1e-12])
    ax.set_yticklabels(['$10^{0}$', '$10^{-4}$', '$10^{-8}$', '$10^{-12}$'], fontsize = fontsize - 2)

    ax.set_xlabel('$L$', fontsize = fontsize, labelpad = 0.)
    ax.set_ylabel('$\mathrm{Error}$', fontsize = fontsize, labelpad = 0.)

    #ax.set_xlabel("$L$", fontsize = fontsize - 2)
    #ax.set_ylabel(r"$\frac{G_{\mathrm{num}} - G_{\mathrm{ana}}}{G_{\mathrm{num}} + G_{\mathrm{ana}}}$", fontsize = fontsize - 2)
    #axInTop.set_ylabel('$\log (N_{\mathrm{pt}})$', fontsize = 7, rotation = 0)
    #axInTop.yaxis.set_label_coords(-0.24, .8)

    #legend = ax.legend(fontsize = fontsize - 2, loc = 'upper left', bbox_to_anchor=(-0.1, 2.), edgecolor = 'black', ncol = 1)
    #legend.get_frame().set_alpha(1.)
    #legend.get_frame().set_boxstyle('Square', pad=0.1)
    #legend.get_frame().set_linewidth(0.0)



    plt.savefig('finiteSize.png', format='png', bbox_inches='tight', dpi = 600)
    #plt.show()


def plotFabry():
    R = 0.3

    def I(x, R):
        X = np.sqrt(R)
        I = 1 / (((1 - X) ** 2) + 4 * X * (np.sin(x)) ** 2)
        return I

    x = np.linspace(2, 10.5, 4000)
    y = 18 * I(x, R)

    fig, ax = plt.subplots(dpi=600)
    fig.set_size_inches(0.8 * 2., 0.8 * 1.)

    for axis in ['top', 'bottom', 'left', 'right']:
        ax.spines[axis].set_linewidth(0.5)

    for axis in ['top', 'right']:
        ax.spines[axis].set_linewidth(0.0)

    # note we must use plt.subplots, not plt.subplot
    # (or if you have an existing figure)
    # fig = plt.gcf()
    # ax = fig.gca()

    ax.vlines(np.pi, 4, 88, "cornflowerblue", linewidth=2)
    ax.text(np.pi + 0.5, 80, r'$\omega_{0}$', fontsize=fontsize)
    ax.plot(x, y, "black", linewidth=0.9)
    ax.set_aspect(0.05)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_yticklabels([])
    ax.set_xticklabels([])
    #ax.axes.xaxis.set_visible(False)
    #ax.axes.yaxis.set_visible(False)
    ax.set_xlim(2., 10.5)
    ax.set_ylim(4., 92)

    ax.set_xlabel('$\omega$', fontsize = fontsize)
    ax.set_ylabel('Transmittance', fontsize = fontsize)

    arrow = patches.FancyArrowPatch((10., 4.), (11.2, 4.), arrowstyle='->', mutation_scale=7, zorder = 100, linewidth=.5, color = 'black', clip_on = False)
    ax.add_patch(arrow)

    arrow = patches.FancyArrowPatch((2., 91.), (2., 93.), arrowstyle='->', mutation_scale=7, zorder = 100, linewidth=.5, color = 'black', clip_on = False)
    ax.add_patch(arrow)

    plt.savefig('fabry.png', format='png', bbox_inches='tight', dpi = 600)


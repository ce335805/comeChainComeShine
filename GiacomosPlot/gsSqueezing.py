from matplotlib import patches
from matplotlib.patches import Ellipse
import numpy as np
from scipy.linalg import cosm
from scipy.sparse.linalg import eigsh
from numpy import arange
import matplotlib.pyplot as plt
import matplotlib.colors as colors
#import matplotlib as mpl
import matplotlib.colors


#mpl.rcParams['font.family'] = 'sans-serif'
#mpl.rcParams['lines.linewidth'] = 2
#mpl.rcParams['lines.markersize'] = 8
#mpl.rcParams['font.size'] = 8  # <-- change fonsize globally
#mpl.rcParams['legend.fontsize'] = 8
#mpl.rcParams['axes.titlesize'] = 8
#mpl.rcParams['axes.labelsize'] = 8
#mpl.rcParams['xtick.major.size'] = 3
#mpl.rcParams['ytick.major.size'] = 3
#mpl.rcParams['xtick.major.width'] = .7
#mpl.rcParams['ytick.major.width'] = .7
#mpl.rcParams['xtick.direction'] = 'out'
#mpl.rcParams['ytick.direction'] = 'out'
#mpl.rcParams['figure.titlesize'] = 8
#mpl.rc('text', usetex = True)
#
##mpl.rcParams['text.latex.preamble'] = [
##        r'\usepackage{helvet}',    # set the normal font here
##        r'\usepackage{sansmath}',  # load up the sansmath so that math -> helvet
##        r'\sansmath'               # <- tricky! -- gotta actually tell tex to use!
##]
#
#mpl.rcParams['text.latex.preamble'] = [
#        r'\usepackage[helvet]{sfmath}']
#
#
#mpl.rcParams['font.family'] = 'sans-serif'
#mpl.rcParams['font.sans-serif'] = 'Helvetica'
#



#mpl.rcParams['font.family'] = 'Helvetica'
#mpl.rcParams['lines.linewidth'] = 2
#mpl.rcParams['lines.markersize'] = 8
#mpl.rcParams['font.size'] = 8  # <-- change fonsize globally
#mpl.rcParams['legend.fontsize'] = 8
#mpl.rcParams['axes.titlesize'] = 8
#mpl.rcParams['axes.labelsize'] = 8
#mpl.rcParams['xtick.major.size'] = 3
#mpl.rcParams['ytick.major.size'] = 3
#mpl.rcParams['xtick.major.width'] = .7
#mpl.rcParams['ytick.major.width'] = .7
#mpl.rcParams['xtick.direction'] = 'out'
#mpl.rcParams['ytick.direction'] = 'out'
#mpl.rcParams['figure.titlesize'] = 8
#mpl.rcParams['text.usetex'] = True
#mpl.rc('text', usetex = True)

#mpl.rcParams['text.latex.preamble'] = [
#       r'\usepackage{helvet}',    # set the normal font here
#       r'\usepackage{sansmath}',  # load up the sansmath so that math -> helvet
#       r'\sansmath'               # <- tricky! -- gotta actually tell tex to use!
#]

#mpl.rcParams['text.latex.preamble'] = [
#    r'\renewcommand{\familydefault}{\sfdefault}',
#    r'\usepackage[scaled=1]{helvet}',
#    r'\usepackage[helvet]{sfmath}',
#    r'\everymath={\sf}'
#]

#mpl.rcParams['font.family'] = 'sans-serif'
#mpl.rcParams['font.sans-serif'] = ["Helvetica"]
#mpl.rcParams['font.sans-serif'] = ['Arial', 'Tahoma', 'DejaVu Sans', 'Lucida Grande', 'Verdana']
#mpl.rcParams['font.family'] = 'sans-serif'
#mpl.rcParams['font.sans-serif'] = 'Helvetica'


#mpl.rcParams['font.family'] = 'sans-serif'
#mpl.rcParams['lines.linewidth'] = 3
#mpl.rcParams['lines.markersize'] = 10
#mpl.rcParams['font.size'] = 20  # <-- change fonsize globally
#mpl.rcParams['legend.fontsize'] = 14
#mpl.rcParams['axes.titlesize'] = 20
#mpl.rcParams['axes.labelsize'] = 20
#mpl.rcParams['xtick.major.size'] = 3
#mpl.rcParams['ytick.major.size'] = 3
#mpl.rcParams['xtick.major.width'] = .7
#mpl.rcParams['ytick.major.width'] = .7
#mpl.rcParams['xtick.direction'] = 'out'
#mpl.rcParams['ytick.direction'] = 'out'
#mpl.rcParams['figure.titlesize'] = 20
##mpl.rcParams['figure.figsize'] = [7.,5]
#mpl.rcParams['text.usetex'] = True



fontsize = 8

def expectation(v, op):
    val = np.real_if_close(np.tensordot(v.conj().T, np.tensordot(op, v, 1), 1))
    return val


def squeezednes_i(g, L, Omega, Nmax):
    B = np.diag(np.sqrt(np.arange((Nmax - 1)) + 1), +1)
    Bd = np.diag(np.sqrt(np.arange((Nmax - 1)) + 1), -1)
    Nb = Bd.dot(B)
    X = B + Bd
    X = B + Bd
    Y = 1j * (Bd - B)
    XX = X.dot(X)
    YY = Y.dot(Y)
    H = -cosm((g / np.sqrt(L)) * X) * (L / np.pi) + (Omega * Nb)
    w, v = eigsh(H, 1, which='SA')
    v = v[:, 0]
    dx = expectation(v, XX)
    dx = dx - (expectation(v, X) ** 2)

    dy = expectation(v, YY)

    dy = dy - (expectation(v, Y) ** 2)

    return np.sqrt(dy / dx)


def Squeezedness_plot(g_0, g_f, Omega1, Omega2, Omega3, L, Omega, Nmax):
    g = np.arange(g_0, g_f + 0.2, .025)
    Ls = [500]

    y01 = []
    for i in range(len(g)):
        y01.append(squeezednes_i(g[i], L, Omega1, Nmax))

    y1 = []
    for i in range(len(g)):
        y1.append(squeezednes_i(g[i], L, Omega2, Nmax))

    y10 = []
    for i in range(len(g)):
        y10.append(squeezednes_i(g[i], L, Omega3, Nmax))

    #fig, ax = plt.subplots(dpi=600)
    gi = [0.2, 0.75]
    ysq = [squeezednes_i(0.2, L, Omega1, Nmax), squeezednes_i(0.75, L, Omega1, Nmax)]
    #fig, ax = plt.subplots(dpi=600)
    #ax.plot(g, y01, color='lightskyblue', label=r'$\omega_{0}= 0.1 t_{h}$', zorder=1)
    #ax.plot(g, y1, color='black', label=r'$\omega_{0}= t_{h}$', zorder=1)
    #ax.plot(g, y10, color='peru', label=r'$\omega_{0}= 10 t_{h}$', zorder=1)
    #ax.scatter(gi, ysq, s=50, marker='x', color='r', linewidths=1.5, zorder=2)

    return [gi, ysq, g, y01, y1, y10]



def Ellypse_squeezing(g, L, Nmax, Omega):
    B = np.diag(np.sqrt(np.arange((Nmax - 1)) + 1), +1)
    Bd = np.diag(np.sqrt(np.arange((Nmax - 1)) + 1), -1)
    Nb = Bd.dot(B)
    X = B + Bd
    X = B + Bd
    Y = 1j * (Bd - B)
    XX = X.dot(X)
    YY = Y.dot(Y)
    H = -cosm((g / np.sqrt(L)) * X) * (L / np.pi) + (Omega * Nb)
    w, v = eigsh(H, 1, which='SA')
    v = v[:, 0]
    dx = expectation(v, XX)
    dx = np.sqrt(dx - (expectation(v, X) ** 2))

    dy = expectation(v, YY)

    dy = np.sqrt(dy - (expectation(v, Y) ** 2))
    Ely = Ellipse((0, 0), dx, dy, color='black', fill=False)

    fig, ax = plt.subplots(dpi=800)
    ax.set_aspect('equal')
    ax.set_xticks([])
    ax.set_yticks([])
    ax.arrow(-0.75 * dx, 0, 1.5 * dx, 0, head_width=0.05, head_length=0.05, fc='k', ec='k')
    ax.arrow(0, -0.75 * dy, 0, 1.5 * dy, head_width=0.05, head_length=0.05, fc='k', ec='k')
    ax.add_patch(Ely)
    return ax


def shaded_ellipse(g, L, Nmax, Omega):
    B = np.diag(np.sqrt(np.arange((Nmax - 1)) + 1), +1)
    Bd = np.diag(np.sqrt(np.arange((Nmax - 1)) + 1), -1)
    Nb = Bd.dot(B)
    X = B + Bd
    Y = 1j * (Bd - B)
    XX = X.dot(X)
    YY = Y.dot(Y)
    H = -cosm((g / np.sqrt(L)) * X) * (L / np.pi) + (Omega * Nb)
    w, v = eigsh(H, 1, which='SA')
    v = v[:, 0]
    dx = expectation(v, XX)
    dx = np.sqrt(dx - (expectation(v, X) ** 2))

    dy = expectation(v, YY)

    dy = np.sqrt(dy - (expectation(v, Y) ** 2))
    sigmax = dx
    sigmay = dy
    dx = 3 * sigmax
    dy = 3 * sigmay

    def z_func(x, y):
        z = (np.exp(-(x ** 2) / (2 * sigmax ** 2))) * (np.exp(-(y ** 2) / (2 * sigmay ** 2)))

        return z

    cnorm = colors.Normalize(-1, 1)
    x = arange(-dx, dx, dx * 0.004)
    y = arange(-dy, dy, dy * 0.004)
    X, Y = np.meshgrid(x, y)  # grid of point
    Z = z_func(X, Y)  # evaluation of the function on the grid

    return [X, Y, Z]

    #fig, ax = plt.subplots(dpi=600)
    #ax.pcolormesh(X, Y, Z, norm=cnorm, cmap='RdBu')
    #ax.set_xticks([])
    #ax.set_yticks([])
    #ax.set_aspect('equal')
    #return ax





def callGiacomosCode():
    g = 3
    L = 1010
    Omega1 = 0.1
    Omega2 = 1
    Omega3 = 10
    g_0 = 0
    g_f = 1
    Nmax = 100
    Omega = .1

    #ax1 = Ellypse_squeezing(g, L, Nmax, Omega)
    #ax2 = Squeezedness_plot(g_0,g_f,Omega1,Omega2,Omega3, L, Omega, Nmax)


    gi, ysq, g, y01, y1, y10 = Squeezedness_plot(g_0,g_f,Omega1,Omega2,Omega3, L, Omega, Nmax)
    print(gi)
    [Xshade1, Yshade1, Zshade1] = shaded_ellipse(gi[0], L, Nmax, Omega)
    [Xshade2, Yshade2, Zshade2] = shaded_ellipse(gi[1], L, Nmax, Omega)


    fig = plt.figure()
    ax = fig.add_subplot(111)
    figWidth = 2.2
    figHeight = 1.8
    fig.set_size_inches(figWidth, figHeight)

    left, bottom, width, height = [0.15, 0.3, 0.3 , 0.3 ]
    axIn1 = fig.add_axes([left, bottom, width, height])
    left, bottom, width, height = [0.29, 0.51, 0.5 , 0.5 ]
    axIn2 = fig.add_axes([left, bottom, width, height])

    for axis in ['top', 'bottom', 'left', 'right']:
        ax.spines[axis].set_linewidth(0.5)
        axIn1.spines[axis].set_linewidth(0.0)
        axIn2.spines[axis].set_linewidth(0.0)

    for axis in ['top', 'right']:
        ax.spines[axis].set_linewidth(0.0)


    axIn1.set_aspect('equal')
    axIn2.set_aspect('equal')

    myblue = '#406080'

    myyellow = '#E6BB65'
    myyellow = '#CD8F14'


    color01 = 'black'
    color1 = myyellow
    color10 = myblue

    ax.plot(g, y01, color=color01, label=r'$\omega_{0}= 0.1 t_{h}$', zorder=1, linewidth = 1.)
    ax.plot(g, y1, color=color1, label=r'$\omega_{0}= t_{h}$', zorder=1, linewidth = 1.)
    ax.plot(g, y10, color=color10, label=r'$\omega_{0}= 10 t_{h}$', zorder=1, linewidth = 1.)
    ax.plot(gi, ysq, marker='x', color='r', zorder=2, markersize = 5., linestyle = '', linewidth = 1.)


    ax.set_xlabel('$g$', fontsize = fontsize, labelpad = -1)
    #ax.set_ylabel(r'$\frac{\Delta P}{\Delta X}$', fontsize = fontsize, labelpad=0., rotation = 0.)
    #ax.yaxis.set_label_coords(-0.12, .8)
    ax.set_ylabel(r"$\frac{\Delta P}{\Delta X}$", fontsize = fontsize, labelpad = 5.)


    ax.set_xticks([0., 0.5, 1.])
    ax.set_xticklabels(['$0$', '$0.5$', '$1$'], fontsize = fontsize)

    ax.set_yticks([1., 2., 3.])
    ax.set_yticklabels(['$1$', '$2$', '$3$'])

    ax.set_xlim(0., 1.05)
    ax.set_ylim(0.9, 3.2)

    plt.gcf().text(.92, 0.83, r"$\underline{\omega_0 \, [t_h]}$", fontsize=fontsize, color='black')
    plt.gcf().text(.92, 0.75, r"$0.1$", fontsize=8, color=color01)
    plt.gcf().text(.92, 0.235, r"$1$", fontsize=8, color=color1)
    plt.gcf().text(.92, 0.14, r"$10$", fontsize=8, color=color10)



    nColors = 1000
    ###make color array
    colorArr = np.zeros((nColors, 4))
    startColor = (.583, 0.7, .5, 1.)
    endColor = (.583, .0, 1., 1.)
    for colorInd, color in enumerate(colorArr):
        linFac = colorInd / (len(colorArr) - 1)
        tempColorHSV = np.array((startColor[0] + linFac * (endColor[0] - startColor[0]), startColor[1] + (endColor[1] - startColor[1]) * linFac, startColor[2] + (endColor[2] - startColor[2]) * linFac))
        colorArr[colorInd, :] = np.append(matplotlib.colors.hsv_to_rgb(tempColorHSV), np.array([startColor[3] + (endColor[3] - startColor[3]) * linFac]))
        #colorArr[colorInd, :] = matplotlib.colors.hsv_to_rgb(tempColorHSV)

    colorArr = np.flip(colorArr, axis = 0)
    myColorMap = matplotlib.colors.ListedColormap(colorArr)

    cnorm = colors.Normalize(1e-2, 1.)
    axIn1.pcolormesh(Xshade1, Yshade1, Zshade1, norm=cnorm, cmap=myColorMap)
    axIn1.set_xticks([])
    axIn1.set_yticks([])

    axIn2.pcolormesh(Xshade2, Yshade2, Zshade2, norm=cnorm, cmap=myColorMap)
    axIn2.set_xticks([])
    axIn2.set_yticks([])

    #axis arrow
    arrow = patches.FancyArrowPatch((0, -3.5), (0, 4), arrowstyle='->', mutation_scale=5, zorder = 100, linewidth=.5, color = 'black', clip_on = False)
    axIn1.add_patch(arrow)
    plt.gcf().text(0.29, 0.63, "$P$", fontsize=fontsize - 2)


    arrow = patches.FancyArrowPatch((-2.5, 0), (3, 0), arrowstyle='->', mutation_scale=5, zorder = 100, linewidth=.5, color = 'black', clip_on = False)
    axIn1.add_patch(arrow)
    plt.gcf().text(0.43, 0.4325, "$X$", fontsize=fontsize - 2)

    #axis arrow
    arrow = patches.FancyArrowPatch((0, -3.5), (0, 4), arrowstyle='->', mutation_scale=5, zorder = 100, linewidth=.5, color = 'black', clip_on = False)
    axIn2.add_patch(arrow)
    plt.gcf().text(0.525, 0.975, "$P$", fontsize=fontsize - 2)


    arrow = patches.FancyArrowPatch((-2., 0), (2.5, 0), arrowstyle='->', mutation_scale=5, zorder = 100, linewidth=.5, color = 'black', clip_on = False)
    axIn2.add_patch(arrow)
    plt.gcf().text(0.67, 0.745, "$X$", fontsize=fontsize - 2)

    #axis arrow
    arrow = patches.FancyArrowPatch((1, 0.9), (1.2, 0.9), arrowstyle='->', mutation_scale=7, zorder = 100, linewidth=.5, color = 'black', clip_on = False)
    ax.add_patch(arrow)

    arrow = patches.FancyArrowPatch((0., 3.), (0., 3.3), arrowstyle='->', mutation_scale=7, zorder = 100, linewidth=.5, color = 'black', clip_on = False)
    ax.add_patch(arrow)


    #plt.show()
    plt.savefig('savedPlots/Fig1d.png', format='png', bbox_inches='tight', dpi = 600)



import numpy as np
import matplotlib.pyplot as plt


def plotSpec(kVec, wVec, spec):

    fig = plt.figure()
    ax = fig.add_subplot(111)
    CS = ax.contourf(kVec, wVec, spec, 1000, cmap = 'gnuplot2')
    fig.colorbar(CS, ax=ax)
    plt.show()

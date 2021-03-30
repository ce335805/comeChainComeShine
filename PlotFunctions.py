import numpy as np
import matplotlib.pyplot as plt

def cosFunc(k):
    return - np.cos(k)


def coscosFunc(k, d):
    denominator = 2. * np.sqrt((np.cos(k)**2 + d * d * np.sin(k)**2))**3
    numerator = - np.cos(k)**4 + d**2 * np.sin(k)**4
    return  2. * (1 - d * d) * numerator / denominator

def plotFunctions(func1, funcArr, paramArray, xArr):
    fig, ax = plt.subplots(nrows=1, ncols=1)
    cmap = plt.cm.get_cmap('terrain')
    for indParam in range(len(paramArray)):
        color = cmap(paramArray[indParam] * 1. / (paramArray[-1] + 0.1))
        ax.plot(xArr, funcArr[indParam, :], color = color, label = "d = {:.2f}".format(paramArray[indParam]))

    ax.plot(xArr, func1, color='r', label='Cos', linestyle='--')
    plt.axhline(y=0.0, color='black', linestyle='-', linewidth = 1)

    plt.legend()
    plt.show()

def compareFunctions(dArr):

    kArr = np.linspace(- np.pi / 2., np.pi / 2., 500, endpoint=False)

    cos = cosFunc(kArr)
    coscos = np.zeros((len(dArr), len(kArr)), dtype='double')
    for indD in range(len(dArr)):
        coscos[indD, :] = coscosFunc(kArr, dArr[indD])

    plotFunctions(cos, coscos, dArr, kArr)


def main():
    print("Let's look at some functions!")

    dArr = np.linspace(0.0, 0.4, 9)
    compareFunctions(dArr)

main()
import globalSystemParams as prms
import arb_order.arbOrder as GSarb
import numpy as np
import comparisonPlots as compPlot
import sec_order.analyticalEGS as secOrder
import energyFunctions as eF
import utils
import greensFunction as green
import beuatifulPlots as bPlots
from automatedTests import gfTests
import fourierTrafo as FT

def main():
    print('The length of the to-be-considered 1D chain is {}'.format(prms.chainLength))

    gfTests.runAllTests()

    etas = np.linspace(0.0, .5, 30)

    #avPhNumbers = GSarb.findPhotonNumbers(etas)
    #avPhNumbersExSec = secOrder.findPhotonNumberExactSec(etas)
    #compPlot.compareArrays(etas, avPhNumbers, avPhNumbers)

    #gsEnergies = GSarb.findGSEnergies(etas)
    #gsEnergiesSecExact = secOrder.findGSEnergyExactSec(etas)
    #compPlot.compareArrays(etas, gsEnergies, gsEnergies)

    damping = .005
    samplesT = 5001 #take an uneven number
    tBound = 50000.
    kVec = np.linspace(0, 2. * np.pi, prms.chainLength)
    tVec = np.linspace(-tBound, tBound , samplesT, endpoint=False)

    eta = .0

    gfT = green.anaGreenVecTComplete(kVec, tVec, eta, damping)
    wVec, gfW = FT.FT(tVec, gfT)
    wVec = np.sort(wVec)
    #compPlot.compareArrays(tVec, np.imag(gfT[25, :]), np.real(gfT[25, :]))
    #compPlot.compareArrays(wVec, np.imag(gfW[25, :]), np.imag(gfW[25, :]))
    bPlots.plotSpec(kVec, wVec, np.transpose(np.imag(gfW[:, :])))

    #damping = .1
    #samplesT = 400
    #tBound = 200.
    #tVec = np.linspace(-tBound, tBound, samplesT)
    #f = np.exp(-1j * (-.5) * tVec - damping * np.abs(tVec))
    #wVec, g = FT.FT(tVec, f)
    ##compPlot.compareArrays(tVec, np.real(f), np.imag(f))
    #compPlot.compareArrays(wVec, np.real(g), np.imag(g))

    print("")
    print("The calculation has finished - Juhu!")

main()


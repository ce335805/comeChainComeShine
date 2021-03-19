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


    kVec = np.linspace(0, 2. * np.pi, prms.chainLength)
    tVec = np.linspace(0., 80. , 100)

    eta = 0.0
    gfNum0 = green.gfNumVecT(kVec, tVec, eta)
    gfT0 = green.anaGreenVecT(kVec, tVec, eta)

    eta = 0.3

    gfNum = green.gfNumVecT(kVec, tVec, eta)
    gfT = green.anaGreenVecT(kVec, tVec, eta)

    #compPlot.compareArrays(tVec, np.imag(gfT0[18, :]), np.imag(gfT[18, :]))
    #compPlot.compareArrays(tVec, np.imag(gfNum0[18, :]), np.imag(gfNum[18, :]))
    compPlot.compareArrays(tVec, np.imag(gfNum[18, :]) - np.imag(gfT[18, :]), np.real(gfNum[18, :]) - np.real(gfT[18, :]))
    #compPlot.compareArrays(tVec, np.imag(gfT0[18, :]) - np.imag(gfT[18, :]), np.real(gfT0[18, :]) - np.real(gfT[18, :]))
    #compPlot.compareArrays(tVec, np.imag(gfNum0[18, :]) - np.imag(gfNum[18, :]), np.real(gfNum0[18, :]) - np.real(gfNum[18, :]))



    print("")
    print("The calculation has finished - Juhu!")

main()


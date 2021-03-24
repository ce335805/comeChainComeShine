import globalSystemParams as prms
import numpy as np
import comparisonPlots as compPlot
from automatedTests import gfTests
from automatedTests import ftTests
import fourierTrafo as FT
from greensFunction import greenAna1st
from greensFunction import greenNum1st
import beuatifulPlots as bPlots


def main():
    print('The length of the to-be-considered 1D chain is {}'.format(prms.chainLength))

    gfTests.runAllTests()
    ftTests.runAllTests()


    #etas = np.linspace(0.0, .5, 30)

    #avPhNumbers = GSarb.findPhotonNumbers(etas)
    #avPhNumbersExSec = secOrder.findPhotonNumberExactSec(etas)
    #compPlot.compareArrays(etas, avPhNumbers, avPhNumbers)

    #gsEnergies = GSarb.findGSEnergies(etas)
    #gsEnergiesSecExact = secOrder.findGSEnergyExactSec(etas)
    #compPlot.compareArrays(etas, gsEnergies, gsEnergies)

    eta = .2
    damping = .001
    kVec = np.linspace(0, 2. * np.pi, prms.chainLength)
    wVec = np.linspace(-0.15, 0.15, 201)

    tVec = FT.tVecFromWVec(wVec)
    tVecPos = tVec[len(tVec)//2 : ]
    #GFTG = greenAna1st.anaGreenVecTGreater(kVec, tVecPos, eta, damping)
    GFWG = greenNum1st.numGreenVecWGreater(kVec, wVec, eta, damping)

    #print("GFWG.shape = {}".format(GFWG.shape))
    #compPlot.compareArrays(wVec, np.imag(GFWG[25, :]), np.real(GFWG[25, :]))
    bPlots.plotSpec(kVec, wVec, np.transpose(np.imag(GFWG)))
    #bPlots.plotSpec(kVec, wVec, np.transpose(np.imag(GFWL)))
    #compPlot.compareArrays(wVec, np.imag(GFWL[5, :]), np.imag(GFWL[12, :]))

    print("")
    print("The calculation has finished - Juhu!")

main()


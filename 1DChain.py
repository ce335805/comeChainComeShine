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
import greenAna1st

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

    eta = 1.1
    damping = .0025
    kVec = np.linspace(0, 2. * np.pi, prms.chainLength)
    wVec = np.linspace(-0.15, 0.15, 2001)

    tVec = FT.tVecFromWVec(wVec)
    tVecPos = tVec[len(tVec)//2 : ]
    GFTG = greenAna1st.anaGreenVecTGreater(kVec, tVecPos, eta, damping)
    #compPlot.compareArrays(tVecPos, np.imag(GFTG[25, :]), np.imag(GFTG[25, :]))

    GFWG = greenAna1st.anaGreenVecWGreater(kVec, wVec, eta, damping)
    #print("GFWG.shape = {}".format(GFWG.shape))
    #compPlot.compareArrays(wVec, np.imag(GFWG[25, :]), np.real(GFWG[25, :]))
    bPlots.plotSpec(kVec, wVec, np.transpose(np.imag(GFWG)))

    #samplesT = 5001 #take an uneven number
    #tBound = 50000.
    #kVec = np.linspace(0, 2. * np.pi, prms.chainLength)
    #tVec = np.linspace(-tBound, tBound , samplesT, endpoint=False)
#
    #eta = .0
#
    #gfT = green.anaGreenVecTComplete(kVec, tVec, eta, damping)
    #wVec, gfW = FT.FT(tVec, gfT)
    #wVec = np.sort(wVec)
    ##compPlot.compareArrays(tVec, np.imag(gfT[25, :]), np.real(gfT[25, :]))
    ##compPlot.compareArrays(wVec, np.imag(gfW[25, :]), np.imag(gfW[25, :]))
    #bPlots.plotSpec(kVec, wVec, np.transpose(np.imag(gfW[:, :])))

    print("")
    print("The calculation has finished - Juhu!")

main()


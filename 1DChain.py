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
import time

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
    tVec = np.linspace(0., 80. , 50)

    eta = 0.3



    print("")
    print("The calculation has finished - Juhu!")

main()


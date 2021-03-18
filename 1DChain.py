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

    gfTests.g1stEQg0()

    etas = np.linspace(0.0, .5, 30)

    #avPhNumbers = GSarb.findPhotonNumbers(etas)
    #avPhNumbersExSec = secOrder.findPhotonNumberExactSec(etas)
    #compPlot.compareArrays(etas, avPhNumbers, avPhNumbers)

    #gsEnergies = GSarb.findGSEnergies(etas)
    #gsEnergiesSecExact = secOrder.findGSEnergyExactSec(etas)
    #compPlot.compareArrays(etas, gsEnergies, gsEnergies)

    eta = 0.0

    kVec = np.linspace(0, 2. * np.pi, prms.chainLength)
    tVec = np.linspace(0., 50. , 100)

    gfNum = green.gfNumVecT(kVec, tVec, eta)

    gfT = green.anaGreenVecT(kVec, tVec, eta)
    #g0T = green.g0VecT(kVec, tVec)
    compPlot.compareArrays(kVec, np.imag(gfNum[:, 23]), np.imag(gfT[:, 23]))

    print("")
    print("The calculation has finished - Juhu!")

main()


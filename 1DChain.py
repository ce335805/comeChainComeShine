import globalSystemParams as prms
import arb_order.arbOrder as GSarb
import numpy as np
import comparisonPlots as compPlot
import sec_order.secOrder as secOrder
import energyFunctions as eF
import utils
import greensFunction as green
import beuatifulPlots as bPlots

def main():
    print('The length of the to-be-considered 1D chain is {}'.format(prms.chainLength))

    etas = np.linspace(0.0, .5, 30)

    #avPhNumbers = GSarb.findPhotonNumbers(etas)
    #avPhNumbersExSec = secOrder.findPhotonNumberExactSec(etas)
    #compPlot.compareArrays(etas, avPhNumbers, avPhNumbers)

    #gsEnergies = GSarb.findGSEnergies(etas)
    #gsEnergiesSecExact = secOrder.findGSEnergyExactSec(etas)
    #compPlot.compareArrays(etas, gsEnergies, gsEnergies)

    eta = 0.1

    kVec = np.linspace(-np.pi, np.pi, prms.chainLength)
    wVec = np.linspace(- (prms.w0 - 3. * prms.t), 30. * prms.w0 - 1. * prms.t , 5000)

    spec = green.calcSpectral(kVec, wVec, eta)
    bPlots.plotSpec(kVec, wVec, spec)

    print("")
    print("The calculation has finished - Juhu!")

main()


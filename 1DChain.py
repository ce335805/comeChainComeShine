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

#    damping = 1e-3
#    samplesT = 1000
#    tBound = 10. / (prms.w0 * damping)
#    kVec = np.linspace(0, 2. * np.pi, prms.chainLength)
#    tVec = np.linspace(- tBound, tBound , samplesT, endpoint=True)
#
#    eta = 0.0
#
#    gfAna = green.anaGreenVecTComplete(kVec, tVec, eta, damping)
#    #compPlot.compareArrays(tVec, np.imag(gfAna[18, :]), np.real(gfAna[18, :]))
#
#    GFW = np.fft.fft(gfAna, norm='ortho')
#    wVec = np.linspace(- np.pi/(2 * tBound / samplesT), np.pi / (2 * tBound / samplesT), samplesT)

    tVec = np.linspace(-20., 20., 200)
    f = np.exp(-1j * 2. * np.pi * 1. * tVec - 0.2 * np.abs(tVec))
    wVec, g = FT.FT(tVec, f)

    #compPlot.compareArrays(tVec, np.real(f), np.imag(f))
    compPlot.compareArrays(wVec, np.real(g), np.imag(g))


    #tBound = 1.
    #NT = 101
    #tVec = np.linspace(0, tBound, NT, endpoint=False)
    #GFT = np.exp(1j * 2. * 50 * tVec[:] - 1. * tVec[:])
#
    #GFW = np.fft.fft(GFT)
    #wVec = np.linspace(0 , NT, NT, endpoint=False)
#
    #compPlot.compareArrays(wVec, np.imag(GFW[:]), np.real(GFW[:]))
    #compPlot.compareArrays(tVec, np.imag(GFT[:]), np.real(GFT[:]))
    #bPlots.plotSpec(kVec[len(kVec)//2 - 10 : len(kVec)//2 + 10], wVec, np.transpose(np.imag(GFW))[:, len(kVec)//2 - 10 : len(kVec)//2 + 10])


    print("")
    print("The calculation has finished - Juhu!")

main()


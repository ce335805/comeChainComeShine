import globalSystemParams as prms
import numpy as np
import comparisonPlots as compPlot
from automatedTests import gfTests
from automatedTests import ftTests
from automatedTests import gsTests
from automatedTests import gsIsEigenstate
from arb_order import arbOrder
import matplotlib.pyplot as plt
from arb_order import photonState
from greensFunction import greenAna1st
from greensFunction import greenNum1st
from greensFunction import greenNumArb
import fourierTrafo as FT
import beuatifulPlots as bPlots

def main():
    print('The length of the to-be-considered 1D chain is {}'.format(prms.chainLength))

    #gfTests.runAllTests()
    ftTests.runAllTests()
    #gsTests.runAllTests()
    #gsIsEigenstate.runAllTests()

    #eta = 0.2

    #initialState = np.zeros(prms.chainLength, dtype='double')
    #initialState[0: prms.numberElectrons] = 1.0
    #gsE = arbOrder.findGS(initialState, eta, 3)

    #fig, ax = plt.subplots(nrows=1, ncols=1)
    #ax.plot(gsE[:])
    #plt.show()

    #phState = photonState.photonGS(gsE, eta, 3)

    #fig, ax = plt.subplots(nrows=1, ncols=1)
    #ax.plot(np.abs(phState[:]))
    #plt.show()



    eta = .2
    damping = .0025
    kVec = np.linspace(0, 2. * np.pi, prms.chainLength)
    wVec = np.linspace(-.15, .15, 1001)

    #GFWGAna = greenAna1st.anaGreenVecWGreater(kVec, wVec, eta, damping)
    #GFWGNum1st = greenNum1st.numGreenVecWGreater(kVec, wVec, eta, damping)
    #GFWGNumArb = greenNumArb.numGreenVecWGreater(kVec, wVec, eta, damping)
    #GFWLAna = greenAna1st.anaGreenVecWLesser(kVec, wVec, eta, damping)
    #GFWLNum = greenNum1st.numGreenVecWLesser(kVec, wVec, eta, damping)

    aGreaterArb = greenNumArb.spectralGreater(kVec, wVec, eta, damping)

    #compPlot.compareArrays(tVecPos, np.imag(GFTGAna[31, :]) - np.imag(GFTGNum[31, :]), np.imag(GFTGAna[31, :]) - np.imag(GFTGNum[31, :]))
    #compPlot.compareArrays(wVec, np.imag(GFWGNum1st[20, :]), np.imag(GFWGNumArb[20, :]))
    bPlots.plotSpec(kVec, wVec, damping * np.transpose(aGreaterArb))

    print("")
    print("The calculation has finished - Juhu!")

main()


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
from fsShift import currentOperator as current
import energyFunctions as eF
from arb_order import photonState as phState
from coherentState import coherentState
from fsShift import gsFromFSShift

def main():
    print('The length of the to-be-considered 1D chain is {}'.format(prms.chainLength))

    gfTests.runAllTests()
    ftTests.runAllTests()
    gsTests.runAllTests()
    gsIsEigenstate.runAllTests()

    eta = .3

    jGauge = current.currentGS(eta)
    print("jGS = {}".format(jGauge))

    #damping = .002
    #kVec = np.linspace(0, 2. * np.pi, prms.chainLength, endpoint=False)
    #wVec = np.linspace(-.3, .3, 2001)

    #aGreaterArb = greenNumArb.spectralGreater(kVec, wVec, eta, damping)
    #aLesserArb = greenNumArb.spectralLesser(kVec, wVec, eta, damping)
    #aArb = aLesserArb
    #bPlots.plotSpecLog(kVec, wVec, damping * np.transpose(aArb))

    #dW = wVec[1] - wVec[0]
    #intArr = dW * np.sum(aArb, axis=1)
    #print(intArr / np.sqrt(2. * np.pi))

    print("")
    print("The calculation has finished - Juhu!")

main()


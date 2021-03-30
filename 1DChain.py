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

def main():
    print('The length of the to-be-considered 1D chain is {}'.format(prms.chainLength))

    #gfTests.runAllTests()
    #ftTests.runAllTests()
    #gsTests.runAllTests()
    #gsIsEigenstate.runAllTests()

    eta = .2

    jGauge = current.currentGS(eta)
    print("jGS = {}".format(jGauge))

    #expAnnihil = current.expectAnnihil(eta)
    #expSin = current.expectSinA(eta)
    #expCos = current.expectCosA(eta)
    #print("<a> = {}".format(expAnnihil))
    #print("<sin(A)> = {}".format(expSin))
    #print("<cos(A)> = {}".format(expCos))

    initialState = np.zeros(prms.chainLength, dtype='double')
    initialState[: prms.numberElectrons // 2 + 1] = 1.0
    initialState[prms.chainLength - prms.numberElectrons // 2 - 0:] = 1.0

    gs = arbOrder.findGS(initialState, eta, 3)
    gsJ = eF.J(gs)
    print("jOp = {}".format(gsJ))
    gsT = eF.T(gs)
    print("tOp = {}".format(gsT))
    ptGS = phState.findPhotonGS([gsT, gsJ], eta, 3)
    #bPlots.plotPtGS(ptGS, eta)
    avPtN = phState.averagePhotonNumber([gsT, gsJ], eta, 3)
    cohState = coherentState.getCoherentStateForN(avPtN)
    bPlots.plotPtGSWithCoh(ptGS, cohState, eta)

    #damping = .002
    #kVec = np.linspace(0, 2. * np.pi, prms.chainLength, endpoint=False)
    #wVec = np.linspace(-.4, .4, 4001)

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


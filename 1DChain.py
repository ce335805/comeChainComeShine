import globalSystemParams as prms
import numpy as np
import comparisonPlots as compPlot
from automatedTests import gfTests
from automatedTests import ftTests
from automatedTests import gsTests
from automatedTests import gsIsEigenstate
from automatedTests import nonEqTests
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
from thermodynamicLimit import photonOccupancies
from thermodynamicLimit import  diagonalizeH
from floquet import spectralFunction
from nonEqGreen import nonEqGreen
from exactGS import exactGS

def main():
    print('The length of the to-be-considered 1D chain is {}'.format(prms.chainLength))

    #gfTests.runAllTests()
    #ftTests.runAllTests()
    #gsTests.runAllTests()
    #gsIsEigenstate.runAllTests()
    #nonEqTests.runAllTests()


    #eta = 3.
    #ptState = exactGS.getExactGS(eta)
    #bPlots.plotPtGS(ptState, eta)
    #xVar = exactGS.xVar(eta)
    #pVar = exactGS.pVar(eta)
    #print("xVar = {}".format(xVar))
    #print("pVar = {}".format(pVar))
    #print("xVar x pVar = {}".format(xVar * pVar))

    eta = .1 / np.sqrt(prms.chainLength)

    damping = .1
    kVec = np.linspace(0, 2. * np.pi, prms.chainLength, endpoint=False)
    wVec = np.linspace(-20., 20., 2000, endpoint=False)
    tAv = np.array([0., 100., 10000., 1000000.])
    tAv = np.linspace(0., 1000., 21)

    gfNonEq = nonEqGreen.gfGSWLesser(kVec, wVec, tAv, eta, damping)
    gfNonEqCoh = nonEqGreen.gfCohWLesser(kVec, wVec, tAv, eta, damping, 10.)

    print("gfNonEq.shape = {}".format(gfNonEq.shape))

    for indTAv in range(len(tAv)):
        compPlot.compareArraysLog(wVec, np.imag(gfNonEq[prms.chainLength // 4, :, indTAv]), np.imag(gfNonEqCoh[prms.chainLength // 4, :, indTAv]))


    #aGreater = greenNumArb.spectralGreater(kVec, wVec, eta, damping)
    #aLesser = - greenNumArb.spectralLesser(kVec, wVec, eta, damping)
    #aLesser1st = - greenNum1st.spectralLesser(kVec, wVec, eta, damping)
    #aLesserFloquet = -spectralFunction.spectralLesser(kVec, wVec, A0, damping)
    #aTotal = - aLesser + aGreater
    #compPlot.compareArrays(wVec, aLesserFloquet[prms.chainLength // 4, :], aLesserFloquet[prms.chainLength // 4, :])
    #bPlots.plotSpecLog(kVec, wVec, damping * np.transpose(np.abs(aLesserFloquet)))

    #dW = wVec[1] - wVec[0]
    #intArr = dW * np.sum(aArb, axis=1)
    #print(intArr / np.sqrt(2. * np.pi))

    print("")
    print("The calculation has finished - Juhu!")

main()


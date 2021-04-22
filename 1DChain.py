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
    nonEqTests.runAllTests()

    #bPlots.calculateAndPlotShakeOffs()

    exit()

    cohN = .3
    eta = .3 / np.sqrt(cohN + 1.)

    tau = 2. * np.pi / prms.w0
    wVec = np.linspace(-10., 10., 500, endpoint=False)
    tAv = np.linspace(0. * tau, 20. * tau, 100, endpoint=True)
    kVecTotal = np.linspace(0, 2. * np.pi, prms.chainLength, endpoint=False)
    kVec = np.array([kVecTotal[prms.chainLength // 4]])
    damping = .1


    gWFloquet = spectralFunction.gLesserW(kVec, wVec, tAv, eta, cohN, damping)
    gWFloquetInt = 1. / (21. * tau) * (tAv[1] - tAv[0]) * np.sum(gWFloquet, axis=2)
    print("gWFloquetInt.shape = {}".format(gWFloquetInt.shape))


    #gfNonEq = nonEqGreen.gfGSWLesser(kVec, wVec, np.array([0.]), eta, damping)
    gfNonEqCoh = nonEqGreen.gfCohWLesser(kVec, wVec, tAv, eta, damping, cohN)
    gfNonEqCohN0 = 1. / (21. * tau) * (tAv[1] - tAv[0]) * np.sum(gfNonEqCoh, axis=2)
    print("gfNonEqCohN0.shape = {}".format(gfNonEqCohN0.shape))


    compPlot.compareArraysLog(wVec, np.imag(gfNonEqCohN0[0, :]), np.imag(gWFloquetInt[0, :]))


    #aGreater = greenNumArb.spectralGreater(kVec, wVec, eta, damping)
    #aLesser = - greenNumArb.spectralLesser(kVec, wVec, eta, damping)
    #aLesser1st = - greenNum1st.spectralLesser(kVec, wVec, eta, damping)
    #aLesserFloquet = -spectralFunction.spectralLesser(kVec, wVec, A0, damping)
    #aTotal = - aLesser + aGreater
    #compPlot.compareArrays(wVec, aLesserFloquet[prms.chainLength // 4, :], aLesserFloquet[prms.chainLength // 4, :])
    #bPlots.plotSpecLog(kVec, wVec, damping * np.transpose(np.abs(aLesserFloquet)))

    print("")
    print("The calculation has finished - Juhu!")

main()


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

    damping = .1
    kVec = np.linspace(0, 2. * np.pi, prms.chainLength, endpoint=False)
    wVec = np.linspace(-10., 10., 2000, endpoint=False)
    tau = 2. * np.pi / prms.w0
    tAv = np.linspace(0, tau, 101)
    A0 = .1
    gWFloquet = spectralFunction.gLesserW(kVec, wVec, tAv, A0, damping)
    gWFloquetInt = 1. / tau * (tAv[1] - tAv[0]) * np.sum(gWFloquet, axis=2)

    compPlot.compareArraysLog(wVec, np.imag(gWFloquetInt[prms.chainLength // 4, :]), np.imag(gWFloquetInt[0, :]))


    #eta = .5 / np.sqrt(prms.chainLength)
    #damping = .1
    #kVec = np.linspace(0, 2. * np.pi, prms.chainLength, endpoint=False)
    #wVec = np.linspace(-10., 10., 1000, endpoint=False)
    ##tAv = np.array([0., 100., 1000.])
    #tau = 2. * np.pi / prms.w0
    #tAv = np.linspace(0. * tau, 100. * tau, 500, endpoint=False)
    #gfNonEq = -nonEqGreen.gfGSWLesser(kVec, wVec, np.array([0.]), eta, damping)
    #gfNonEqCoh = -nonEqGreen.gfCohWLesser(kVec, wVec, tAv, eta, damping, 10.)
    #gfNonEqCohN0 = 1. / tau * (tAv[1] - tAv[0]) * np.sum(gfNonEqCoh, axis=2)


    #for indTAv in range(len(tAv)):
    #    compPlot.compareArraysLog(wVec, np.imag(gfNonEq[prms.chainLength // 4, :, indTAv]), np.imag(gfNonEqCoh[prms.chainLength // 4, :, indTAv]))
    #compPlot.compareArraysLog(wVec, np.imag(gfNonEq[prms.chainLength // 4, :, 0]), np.imag(gfNonEqCohN0[prms.chainLength // 4, :]))


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


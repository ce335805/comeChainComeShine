import arb_order.arbOrder
import beuatifulPlots
import globalSystemParams as prms
import numpy as np
import h5py
import comparisonPlots as compPlot
from automatedTests import gfTests
from automatedTests import ftTests
from automatedTests import gsTests
from automatedTests import gsIsEigenstate
from automatedTests import matrixDiagonalization
from automatedTests import nonEqTests
from automatedTests import floquetTests
from arb_order import arbOrder
import matplotlib.pyplot as plt
from arb_order import photonState
from greensFunction import greenAna1st
from greensFunction import greenAna2nd
from greensFunction import greenNum1st
from greensFunction import greenNumArb
import fourierTrafo as FT
import beuatifulPlots as bPlots
from fsShift import currentOperator as current
import energyFunctions as eF
from arb_order import photonState as phState
from coherentState import coherentState
from fsShift import gsFromFSShift
from greensFunction.greenNum1st import gfNumVecTLesser, numGreenVecWLesser, numGreenVecWGreater
from thermodynamicLimit import photonOccupancies
from thermodynamicLimit import  diagonalizeH
from floquet import spectralFunction
from nonEqGreen import nonEqGreen
from exactGS import exactGS
from multiProcGreen import greenKArr
from multiProcGreen import floquetKArr
from fileHandling import writeGreenToFile
from fileHandling import readGreenFromFile
from finiteSizeScale import gfError


def main():
    print('The length of the to-be-considered 1D chain is {}'.format(prms.chainLength))

    #matrixDiagonalization.runAllTests()
    #gfTests.runAllTests()
    #ftTests.runAllTests()
    #gsTests.runAllTests()
    #gsIsEigenstate.runAllTests()
    #nonEqTests.runAllTests()
    #floquetTests.runAllTests()
    #exit()

    #eta = 1. / np.sqrt(prms.chainLength)
    #gsJ = 0.
    #gs = np.zeros((prms.chainLength))
    #gs[0: prms.numberElectrons // 2 + 1] = 1.
    #gs[- prms.numberElectrons // 2 + 1:] = 1.
    #kVec = np.linspace(-np.pi, np.pi, prms.chainLength, endpoint=False)
    #gsT = np.sum(-2. * prms.t * np.cos(kVec) * gs)
    #phGS = phState.findPhotonGS([gsT, gsJ], eta, 3)
    #nAv = photonState.averagePhotonNumber([gsJ, gsT], eta, 3)
    #print("nAv = {}".format(nAv))
    #bPlots.plotPtGSWithCoh(phGS, nAv, eta, gsT)
    #exit()


    #etasNonNorm = np.linspace(0., 1.5, 7, endpoint = True)
    #beuatifulPlots.plotOccsLs(etasNonNorm, 2)
    #exit()

    #etas = np.linspace(0., 2., 7, endpoint=True) / np.sqrt(prms.chainLength)
    #beuatifulPlots.plotLandscapesAllOrders(etas, 3)
    #beuatifulPlots.plotLandscapes1Order(etas, 1)
    #beuatifulPlots.plotLandscapes2Order(etas, 1)
    #exit()

    etaNonNorm = 1.
    Ls = np.array([210, 310, 410, 510, 610, 710, 810, 910, 1010, 1210, 1410, 1610, 2010])

    gfError.gfErrorForLs(etaNonNorm, Ls)
    meanErr = gfError.getMeanErrors(etaNonNorm, Ls)
    maxErr = gfError.getMaxErrors(etaNonNorm, Ls)

    errA1Mean = meanErr[0]
    errA2Mean = meanErr[1]
    errA1Max = maxErr[0]
    errA2Max = maxErr[1]

    print(errA1Mean)
    print(errA1Max)
    print(errA2Mean)
    print(errA2Max)
    compPlot.finiteSizeErrors(Ls, errA1Mean, errA1Max, errA2Mean, errA2Max)

    exit()


    eta1 = 1. / np.sqrt(prms.chainLength)
    eta2 = 0.1 / np.sqrt(prms.chainLength)
    bPlots.plotAnalyticalConductivity(eta1, eta2, 0.)
    #bPlots.plotAnalyticalConductivityImaginary(eta1, eta2, 0.)
    exit()

    #calculate Green's function


    damping = 0.025
    eta = 1. / np.sqrt(prms.chainLength)
    kVec = np.linspace(-np.pi, np.pi, prms.chainLength, endpoint=False)
#    wVec = np.linspace(-8, 8, 8000, endpoint=False)
    wVec = np.linspace(-8, 8, 8000, endpoint=False)
    gAna2W = greenAna2nd.anaGreenVecW(kVec, wVec, eta, damping)
    #writeGreenToFile.writeGreen("data/eqGreen.h5", "gfEq", gAna2W)
    gAna2W = readGreenFromFile.readGreen("data/eqGreen.h5", "gfEq")
    bPlots.plotSpecLog(wVec, 1. / np.sqrt(2. * np.pi) * np.imag(np.transpose(gAna2W)), eta)

    #greenNum1 = greenNum1st.spectralGreater(kVec, wVec, eta, damping)
    #greenAna1 = greenAna1st.spectralGreater(kVec, wVec, eta, damping)
    #compPlot.compareArraysLog(wVec, greenNum1[0, :], greenAna1[0, :])

    exit()


    eta = 2. / np.sqrt(prms.chainLength)
    tau = 2. * np.pi / prms.w0
    wVec = np.linspace(-4., 4., 2000, endpoint=False)
    tAv = np.linspace(0. * tau, 1. * tau, 100, endpoint=False)
    kVec = np.linspace(-np.pi, np.pi, 17, endpoint=True)
    damping = .05
#
#    gWFloquet = floquetKArr.floquetGreenMultiProc(kVec, wVec, tAv, eta, damping, 2)
#    gWFloquetInt = 1. / (5 * tau) * (tAv[1] - tAv[0]) * np.sum(gWFloquet, axis=2)
#    bPlots.greenWaterFallOnlyFloquet(kVec, wVec, gWFloquetInt)
#
#    exit()


    LArr = np.array([102, 102])

    gfFloq = readGreenFromFile.readGreen("data/floquetGreenJ8.h5", "gfFloquet")
    gfArr = readGreenFromFile.readGreen("data/nonEqGreenJ8.h5", "gfNonEq")
    print("gfFloquet.shape = {}".format(gfFloq.shape))
    print("gfArr.shape = {}".format(gfArr.shape))
    bPlots.greenWaterFall(kVec, wVec, gfArr, LArr, gfFloq, .1)
    exit()

    gfArr = np.zeros((len(LArr), len(kVec), len(wVec)),dtype=complex)
    gfFloq = np.zeros((len(kVec), len(wVec)),dtype=complex)
    for lInd, lVal in enumerate(LArr):
        prms.chainLength = lVal
        prms.numberElectrons = lVal // 2
        eta = 0.
        cohN = 0.
        if(lInd ==0) :
            cohN = 3.
            eta = .1 / np.sqrt(lVal)
        else :
            cohN = .3
            eta = .5 / np.sqrt(lVal)
        prms.maxPhotonNumber = 10

        gfNonEq = greenKArr.nonEqGreenMultiProc(kVec, wVec, tAv, eta, damping, cohN)
        gfNonEqN0 = 1. / (21. * tau) * (tAv[1] - tAv[0]) * np.sum(gfNonEq, axis=2)

        gfArr[lInd, :, :] = gfNonEqN0

        if(lInd == len(LArr) - 1):
            gWFloquet = floquetKArr.floquetGreenMultiProc(kVec, wVec, tAv, eta, damping, cohN)
            gWFloquetInt = 1. / (21. * tau) * (tAv[1] - tAv[0]) * np.sum(gWFloquet, axis=2)
            gfFloq = gWFloquetInt


    writeGreenToFile.writeGreen("data/floquetGreen", "gfFloquet", gfFloq)
    writeGreenToFile.writeGreen("data/nonEqGreen", "gfNonEq", gfArr)

    bPlots.greenWaterFall(kVec, wVec, gfArr, LArr, gfFloq, eta)


    #just a check for the sign of the squeezing transformation
    #N = 1
    #T = prms.t / (np.pi) * (np.sin(np.pi / 2.) - np.sin(-np.pi / 2.)) * prms.chainLength
    #J = prms.t / (np.pi) * (np.cos(np.pi / 2.) - np.cos(-np.pi / 2.)) * prms.chainLength
    #eta = 1. / np.sqrt(prms.chainLength)
    #ptGS = photonState.findPhotonGS([T, J], eta, 2)
    #beuatifulPlots.plotPtGSWithCoh(ptGS, N, eta, T)

    print("")
    print("The calculation has finished - Juhu!")

main()


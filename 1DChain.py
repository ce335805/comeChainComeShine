import arb_order.arbOrder
import beuatifulPlots
import globalSystemParams as prms
import numpy as np
import h5py
import comparisonPlots as compPlot
import multiProcGreen
import utils
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
from conductivity import calcConductivity
from multiProcGreen import gfNonEqMultiParam
import arb_order.photonState as ptState


from GiacomosPlot import gsSqueezing

def main():
    print('The length of the to-be-considered 1D chain is {}'.format(prms.chainLength))

    #gsSqueezing.callGiacomosCode()
    #exit()


    ###convergence in boson cutoff
    #eta = 1./np.sqrt(prms.chainLength)
#
    #prms.reuseSin = utils.calcSinAdaggerA(eta)
    #prms.reuseCos = utils.calcCosAdaggerA(eta)
#
    ##prms.maxPhotonNumber = 100
    #gs = arbOrder.findGS(eta, 3)
    #print(gs)
    #print("gs particle Number difference = {}".format(utils.electronNumberZero(gs)))
    #gsJ = eF.J(gs)
    #print("gsJ = {}".format(gsJ))
    #print("gsT = {}".format(eF.T(gs)))
#
    #trueGS = np.zeros(prms.chainLength, dtype='double')
    #trueGS[ : prms.numberElectrons // 2 + 1] = 1.0
    #trueGS[ prms.chainLength - prms.numberElectrons//2: ] = 1.0
#
    #gsEnergy = ptState.energyFromState(trueGS, eta, 3)
    #calculatedGSEnergy = ptState.energyFromState(gs, eta, 3)
#
    #print("energy - gsEnergy = {}".format(calculatedGSEnergy - gsEnergy))
#
    #exit()
#
    #gsT = eF.T(gs)
    #GS200 = phState.findPhotonGS([gsT, gsJ], eta, 3)
    #GS200 = GS200 * GS200
    #nPhot200 = photonState.averagePhotonNumber([gsT, gsJ], eta, 3)
#
    ##compareCutoffs = np.array([10, 20, 30, 50, 70, 100, 150])
    #compareCutoffs = np.array([4, 8, 16, 20, 30, 40, 50, 60, 80, 100, 120, 150, 180]) + 1
    #errNPhots = np.zeros((len(compareCutoffs)))
    #maxErr = np.zeros((len(compareCutoffs)))
    #relErrn0 = np.zeros((len(compareCutoffs)))
    #relErrn2 = np.zeros((len(compareCutoffs)))
    #relErrn4 = np.zeros((len(compareCutoffs)))
    #relErrn8 = np.zeros((len(compareCutoffs)))
    #relErrn16 = np.zeros((len(compareCutoffs)))
    #relErrn32 = np.zeros((len(compareCutoffs)))
#
    #for nMaxInd, nMax in enumerate(compareCutoffs):
    #    prms.maxPhotonNumber = nMax
    #    gs = arbOrder.findGS(eta, 3)
    #    gsJ = eF.J(gs)
    #    print("gsJ = {}".format(gsJ))
    #    gsT = eF.T(gs)
    #    nPhotN = photonState.averagePhotonNumber([gsT, gsJ], eta, 3)
    #    errNPhots[nMaxInd] =2. *  (nPhotN - nPhot200) / (nPhotN + nPhot200)
    #    GSN = phState.findPhotonGS([gsT, gsJ], eta, 3)
    #    GSN = GSN * GSN
    #    compareCutOff = np.amin([nMax, 31])
    #    maxRelErr = 2. * np.amax((GSN[:compareCutOff:2] - GS200[:compareCutOff:2]) / (GSN[:compareCutOff:2] + GS200[:compareCutOff:2]))
    #    #maxRelErr = .5 * np.amax((GSN[::2] - GS200[:nMax:2]) / (GSN[::2] + GS200[:nMax:2]))
    #    maxErr[nMaxInd] = maxRelErr
    #    relErrn0[nMaxInd] = 2 * np.abs((GSN[0] - GS200[0]) / (GSN[0] + GS200[0]))
    #    relErrn2[nMaxInd] = 2 * np.abs((GSN[2] - GS200[2]) / (GSN[2] + GS200[2]))
    #    relErrn4[nMaxInd] = 2 * np.abs((GSN[4] - GS200[4]) / (GSN[4] + GS200[4]))
    #    if(nMax < 9): continue
    #    relErrn8[nMaxInd] = 2 * np.abs((GSN[8] - GS200[8]) / (GSN[8] + GS200[8]))
    #    if(nMax < 17): continue
    #    relErrn16[nMaxInd] = 2 * np.abs((GSN[16] - GS200[16]) / (GSN[16] + GS200[16]))
    #    if(nMax < 33): continue
    #    relErrn32[nMaxInd] = 2 * np.abs((GSN[16] - GS200[16]) / (GSN[16] + GS200[16]))
    #    #print(GSN[:6:2])
#
    #bPlots.finiteConvergenceSizeHilbert(compareCutoffs, np.abs(errNPhots), np.abs(relErrn0), np.abs(relErrn2), np.abs(relErrn4), np.abs(relErrn8), np.abs(relErrn16), np.abs(relErrn32))
#
#
    #print(errNPhots)
    #print(maxErr)
#
    #exit()



    ###compare to Giacomos ED result
    #eta = 1./np.sqrt(prms.chainLength)
#
    #edGS = np.load('data/check_1.npy')
    #print(edGS)
    #print(np.sum(edGS))
#
    #gs = arbOrder.findGS(eta, 3)
    #gsJ = eF.J(gs)
    ##print("gsJ = {}".format(gsJ))
    #gsT = eF.T(gs)
    ##print("gsT = {}".format(gsT))
    ##print("gsT continuous = {}".format(4. * prms.t * prms.numberElectrons / np.pi))
    #varGS = phState.findPhotonGS([gsT, gsJ], eta, 3)
    #varGS = varGS[:21]
    #varGS = varGS * varGS
#
    #print(varGS)
    #print(varGS - edGS)
    ##bPlots.plotComparisonWithED(edGS, varGS, eta, gsT)
    ##bPlots.plotDifferencesToED(np.abs(edGS - varGS), eta, gsT)
    #exit()

    #matrixDiagonalization.runAllTests()
    #gfTests.runAllTests()
    #ftTests.runAllTests()
    #gsTests.runAllTests()
    #gsIsEigenstate.runAllTests()
    #nonEqTests.runAllTests()
    #floquetTests.runAllTests()
    #exit()

    #beuatifulPlots.plotFabry()
    #exit()

    #beuatifulPlots.plotShiftInsetes()
    #exit()

    #beuatifulPlots.arbitraryEDist()
    #exit()


    #origL = prms.chainLength
    #prms.chainLength = 510
    #etaLong = 2. / np.sqrt(prms.chainLength)
    #prms.reuseSin = utils.calcSinAdaggerA(etaLong)
    #prms.reuseCos = utils.calcCosAdaggerA(etaLong)
    #prms.numberElectrons = prms.chainLength // 2
    ##gs = np.zeros((prms.chainLength))
    ##gs[0: prms.numberElectrons // 2 + 1] = 1.
    ##gs[- prms.numberElectrons // 2 + 1:] = 1.
    #gs = arbOrder.findGS(etaLong, 3)
    #gsJ = eF.J(gs)
    #gsTLong = eF.T(gs)
    #print("gsJ = {}".format(gsJ))
    #phGSL = phState.findPhotonGS([gsTLong, gsJ], etaLong, 3)
#
    #prms.chainLength = 10
    #eta = 2. / np.sqrt(prms.chainLength)
    #prms.reuseSin = utils.calcSinAdaggerA(eta)
    #prms.reuseCos = utils.calcCosAdaggerA(eta)
    #prms.numberElectrons = prms.chainLength // 2
    ##gs = np.zeros((prms.chainLength))
    ##gs[0: prms.numberElectrons // 2 + 1] = 1.
    ##gs[- prms.numberElectrons // 2 + 1:] = 1.
    #gs = arbOrder.findGS(eta, 3)
    #gsJ = eF.J(gs)
    #gsT = eF.T(gs)
    #print("gsJ = {}".format(gsJ))
    #phGSS = phState.findPhotonGS([gsT, gsJ], eta, 3)
#
    #bPlots.plotPtGSWithCoh(phGSL, phGSS, etaLong, gsTLong)
#
    #prms.chainLength = origL
    #exit()


    #etasNonNorm = np.linspace(0., 1.5, 7, endpoint = True)
    #beuatifulPlots.plotOccsLs(etasNonNorm, 2)
    #exit()

    #etas = np.linspace(0., 2., 7, endpoint=True) / np.sqrt(prms.chainLength)
    #etas[3] = np.sqrt((np.pi) / (4. * prms.chainLength)) + 0.001 / np.sqrt(prms.chainLength)
    #beuatifulPlots.plotLandscapesAllOrders(etas, 3)
    #beuatifulPlots.plotLandscapes1Order(etas, 1)
    #beuatifulPlots.plotLandscapes2Order(etas, 2)
    #exit()

#    etaNonNorm = 1.
#    ##Ls = np.array([90, 110, 210, 310, 410, 610, 810, 1010, 1410, 1810, 2210, 3010, 4010, 5010, 7010, 10010])
#    #Ls = np.array([90, 110, 210, 310, 410, 610, 810, 1010, 5010, 10010, 15010, 20010, 50010, 100010, 500010, 1000010])
#    Ls = np.array([610, 810, 1010, 5010, 10010, 15010, 20010, 50010, 100010, 500010, 1000010])
#    #gfError.gfErrorForLs(etaNonNorm, Ls)
#    #exit()
#    meanErr = gfError.getMeanErrors(etaNonNorm, Ls)
#    maxErr = gfError.getMaxErrors(etaNonNorm, Ls)
#    meanErr0 = gfError.getMeanErrors0(etaNonNorm, Ls)
#    maxErr0 = gfError.getMaxErrors0(etaNonNorm, Ls)
#
#    beuatifulPlots.finiteSizeErrors(Ls, meanErr, maxErr, meanErr0, maxErr0)
#    #compPlot.finiteSizeErrors(Ls, errA2Mean, errA2Max)
#    exit()


    #eta = 1. / np.sqrt(prms.chainLength)
    #gsT = - 2. / np.pi * prms.chainLength
    #gsKineticAna = coherentState.gsEffectiveKineticEnergy(eta)
    #gsKintic = calcConductivity.expectationCos(eta) * gsT / prms.chainLength
    #print(gsKineticAna)
    #print("")
    #print(gsKintic)
    #exit()

    #eta1 = 1. / np.sqrt(prms.chainLength)
    #eta2 = 0.3 / np.sqrt(prms.chainLength)
    #bPlots.plotAnalyticalConductivity(eta1, eta2, 0.)
    #bPlots.plotAnalyticalConductivityImaginary(eta1, eta2, 0.)
    #delta = 0.02
    #wVec = np.linspace(-30., 30., 30000, endpoint = False)
    #condAna = calcConductivity.calcConductivityAna(wVec, delta, eta1)
    #condNum = calcConductivity.calcConductivityNum(wVec, delta, eta1)
    #compPlot.compareArrays(wVec, np.real(condNum), np.real(condAna))
    #compPlot.compareArrays(wVec, np.imag(condNum), np.imag(condAna))

    #exit()

    #calculate Green's function


    #damping = 0.025
    #eta = 1. / np.sqrt(prms.chainLength)
    #kVec = np.linspace(0, 2. * np.pi, prms.chainLength, endpoint=False)
    #wVec = np.linspace(-8, 8, 8000, endpoint=False)
    ##gAna2W = greenAna2nd.anaGreenVecW(kVec, wVec, eta, damping)
    ##gfNumInf = greenNumArb.numGreenVecWGreater(kVec, wVec, eta, damping) + greenNumArb.numGreenVecWLesser(kVec, wVec, eta, damping)
    ##GF = gfNumInf
    ##writeGreenToFile.writeGreen("data/eqGreenNum.h5", "gfEq", GF)
    #GF = readGreenFromFile.readGreen("clusterData/eqGreenNumN50.h5", "gfEq")
    #bPlots.plotSpecLogDashed(wVec, 1. / np.sqrt(2. * np.pi) * np.imag(np.transpose(GF)), eta)
##
    #exit()

    #greenNum1 = greenNum1st.spectralGreater(kVec, wVec, eta, damping)
    #greenAna1 = greenAna1st.spectralGreater(kVec, wVec, eta, damping)
    #compPlot.compareArraysLog(wVec, greenNum1[0, :], greenAna1[0, :])

#    exit()
#
    #eta = 2. / np.sqrt(prms.chainLength)
    wVec = np.linspace(-5., 5., 2000, endpoint=False)
    kPoint = 3. / 8. * np.pi
    kVec = np.array([kPoint])
    damping = .025

    nArr = np.logspace(np.log10(0.4 / 2.5**2), np.log10(30.), 20, endpoint=True)
    nArr = nArr[1:]#exclude lowest value and replace by GS
    nArr = np.append(np.array([0.]), nArr)
    etaArrNoGS = np.sqrt(.4 / nArr[1:]) / np.sqrt(prms.chainLength)
    etaArr = np.append(np.array([2.5 / np.sqrt(prms.chainLength)]), etaArrNoGS)
    #etaArr = np.ones(len(nArr)) * .5 / np.sqrt(prms.chainLength)

    print('nArr = {}'.format(nArr))

    print('etaArr = {}'.format(etaArr * np.sqrt(prms.chainLength)))
    assert(len(etaArr) == len(nArr))

    #gfArr = gfNonEqMultiParam.nonEqGreenMultiParamMultiProc(kPoint, wVec, damping, nArr)
    #writeGreenToFile.writeGreen("data/nonEqGreenEtaMany", "gfNonEq", gfArr)

    #tau = 2. * np.pi / prms.w0
    #tAv = np.linspace(- .5 * tau, .5  * tau, 20, endpoint=False)
    #gWFloquet = floquetKArr.floquetGreenMultiProc(kVec, wVec, tAv, etaArr[-1], damping, nArr[-1])
    #gWFloquetInt = 1. / tau * (tAv[1] - tAv[0]) * np.sum(gWFloquet, axis=2)
    #gfFloq = gWFloquetInt
    #writeGreenToFile.writeGreen("data/floquetGreen", "gfFloquet", gfFloq)

    gfFloq = readGreenFromFile.readGreen("clusterData/floquetGreenN130Constg", "gfFloquet")
    gfArr = readGreenFromFile.readGreen("clusterData/nonEqGreenEtaManyConstgN130", "gfNonEq")
    print("gfArr.shape = {}".format(gfArr.shape))
    #bPlots.quantumToFloquetCrossover(wVec, 1. / np.sqrt(2. * np.pi) * gfArr, 1. / np.sqrt(2. * np.pi) * gfFloq[0, :], etaArr, nArr)
    bPlots.quantumToFloquetCrossoverConstg(wVec, 1. / np.sqrt(2. * np.pi) * gfArr, 1. / np.sqrt(2. * np.pi) * gfFloq[0, :], etaArr, nArr)
    exit()

    #    for etaInd, eta in enumerate(etaArr):
#        gs = arbOrder.findGS(eta, 3)
#        gsT = eF.T(gs)
#        wTilde = np.sqrt(1. - 2. * eta**2 / prms.w0 * gsT)
#        tau = 2. * np.pi / wTilde
#        tauLength = 1.
#        tAv = np.linspace(200. * tau, (200. + tauLength) * tau, 200, endpoint=False)
#
#        gfNonEq = greenKArr.nonEqGreenMultiProc(kVec, wVec, tAv, etaArr[etaInd], damping, nArr[etaInd])
#        print('gfNonEq.shape = {}'.format(gfNonEq.shape))
#        gfNonEqN0 = 1. / (tauLength * tau) * (tAv[1] - tAv[0]) * np.sum(gfNonEq, axis=2)
#
#        gfArr[etaInd, :, :] = gfNonEqN0
#
#        #if(lInd == len(LArr) - 1):
#        #    gWFloquet = floquetKArr.floquetGreenMultiProc(kVec, wVec, tAv, eta, damping, cohN)
#        #    gWFloquetInt = 1. / (tauLength * tau) * (tAv[1] - tAv[0]) * np.sum(gWFloquet, axis=2)
#        #    gfFloq = gWFloquetInt


#    writeGreenToFile.writeGreen("data/floquetGreen", "gfFloquet", gfFloq)
#    writeGreenToFile.writeGreen("data/nonEqGreenEta", "gfNonEq", gfArr)

    #bPlots.greenWaterFall(kVec, wVec, gfArr, etaArr, gfFloq, eta)
    #bPlots.quantumToFloquetCrossover(wVec, gfArr, gfFloq, etaArr, nArr)

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

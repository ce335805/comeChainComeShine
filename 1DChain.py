import arb_order.arbOrder
import beuatifulPlots
import globalSystemParams as prms
import numpy as np
import h5py
import comparisonPlots as compPlot
import multiProcGreen
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

def main():
    print('The length of the to-be-considered 1D chain is {}'.format(prms.chainLength))


    #eta  = 1. / np.sqrt(prms.chainLength)
    #numCos = calcConductivity.expectationCos(eta)
    #gsT = - 2. / np.pi * prms.chainLength
    #wTilde = np.sqrt(1 - 2. * eta**2 / prms.w0 * gsT)
    #anaCos = 1. - eta**2 * prms.w0 / (2. * wTilde)
    #print("numCos = {}".format(numCos))
    #print("anacos = {}".format(anaCos))
    #wVec = np.linspace(-250., 250., 2000, endpoint=False)
    #tVec = FT.tVecFromWVec(wVec)
    #damping = 0.025
    #expSinSinT = calcConductivity.expectationSinSin(tVec, eta)
    #expSinSinAna = eta**2 * prms.w0 / wTilde * np.exp(-1j * wTilde * tVec)
    #compPlot.compareArrays(tVec, expSinSinT - expSinSinAna, expSinSinAna - expSinSinAna)
    #exit()

    #matrixDiagonalization.runAllTests()
    #gfTests.runAllTests()
    #ftTests.runAllTests()
    #gsTests.runAllTests()
    #gsIsEigenstate.runAllTests()
    #nonEqTests.runAllTests()
    #floquetTests.runAllTests()
    #exit()

    beuatifulPlots.arbitraryEDist()
    exit()

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

    #etaNonNorm = 1.
    ##Ls = np.array([90, 110, 210, 310, 410, 610, 810, 1010, 1410, 1810, 2210, 3010, 4010, 5010, 7010, 10010])
    #Ls = np.array([90, 110, 210, 310, 410, 610, 810, 1010, 5010, 10010, 15010, 20010, 50010, 100010, 500010, 1000010])
    ##gfError.gfErrorForLs(etaNonNorm, Ls)
    #meanErr = gfError.getMeanErrors(etaNonNorm, Ls)
    #maxErr = gfError.getMaxErrors(etaNonNorm, Ls)
    ##errA1Mean = meanErr[0]
    #errA2Mean = meanErr
    ##errA1Max = maxErr[0]
    #errA2Max = maxErr
    ##print(errA1Mean)
    ##print(errA1Max)
    #print(errA2Mean)
    #print(errA2Max)
    #compPlot.finiteSizeErrors(Ls, errA2Mean, errA2Max)
    #exit()

    #gs = arbOrder.findGS(eta, 3)
    #gsJ = eF.J(gs)
    #gsT = eF.T(gs)
    #gsJ = 0.
    #gsT = - 2. / np.pi * prms.chainLength
    #eta = 1. / np.sqrt(prms.chainLength)
    #phGS = phState.findPhotonGS([gsT, gsJ], eta, 3)
    #phGSAna = coherentState.getSqueezedState(eta, gsT)
    #print(phGS)
    #print("")
    #print(phGSAna)
    #exit()

    #eta = 1. / np.sqrt(prms.chainLength)
    #gsT = - 2. / np.pi * prms.chainLength
    #gsKineticAna = coherentState.gsEffectiveKineticEnergy(eta)
    #gsKintic = calcConductivity.expectationCos(eta) * gsT / prms.chainLength
    #print(gsKineticAna)
    #print("")
    #print(gsKintic)
    #exit()

    eta1 = 1. / np.sqrt(prms.chainLength)
    eta2 = 0.1 / np.sqrt(prms.chainLength)
    bPlots.plotAnalyticalConductivity(eta1, eta2, 0.)
    bPlots.plotAnalyticalConductivityImaginary(eta1, eta2, 0.)
    #delta = 0.02
    #wVec = np.linspace(-30., 30., 30000, endpoint = False)
    #condAna = calcConductivity.calcConductivityAna(wVec, delta, eta1)
    #condNum = calcConductivity.calcConductivityNum(wVec, delta, eta1)
    #compPlot.compareArrays(wVec, np.real(condNum), np.real(condAna))
    #compPlot.compareArrays(wVec, np.imag(condNum), np.imag(condAna))

    exit()

    #calculate Green's function


#    damping = 0.025
#    eta = 1. / np.sqrt(prms.chainLength)
#    kVec = np.linspace(0, 2. * np.pi, prms.chainLength, endpoint=False)
#    wVec = np.linspace(-8, 8, 8000, endpoint=False)
#    #gAna2W = greenAna2nd.anaGreenVecW(kVec, wVec, eta, damping)
#    #gfNumInf = greenNumArb.numGreenVecWGreater(kVec, wVec, eta, damping) + greenNumArb.numGreenVecWLesser(kVec, wVec, eta, damping)
#    #GF = gfNumInf
#    #writeGreenToFile.writeGreen("data/eqGreenNum.h5", "gfEq", GF)
#    GF = readGreenFromFile.readGreen("data/eqGreenNum.h5", "gfEq")
#    bPlots.plotSpecLogDashed(wVec, 1. / np.sqrt(2. * np.pi) * np.imag(np.transpose(GF)), eta)
#
#    exit()

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

#    gWFloquet = floquetKArr.floquetGreenMultiProc(kVec, wVec, tAv, eta, damping, 2)
#    gWFloquetInt = 1. / (5 * tau) * (tAv[1] - tAv[0]) * np.sum(gWFloquet, axis=2)
#    bPlots.greenWaterFallOnlyFloquet(kVec, wVec, gWFloquetInt)
#    exit()


#    etaArr = np.array([2.5, 2., np.sqrt(2.), 1., 1./np.sqrt(2.), 0.5]) / np.sqrt(prms.chainLength)
#    nArr = np.array([0., 0.1, 0.2, 0.4, 0.8, 1.6])
#    assert(len(etaArr) == len(nArr))


    etaArr = np.array([2.5, 2., 0.5]) / np.sqrt(prms.chainLength)
    nArr = np.array([0., 0.1, 0.4, 0.8, 3.2, 12.8])
    nArr = np.logspace(np.log10(0.4 / 2.5**2), np.log10(30.), 20, endpoint=True)
    #nArr = np.logspace(np.log10(15), np.log10(20.), 2, endpoint=True)
    nArr = nArr[1:]#exclude lowest value and replace by GS
    nArr = np.append(np.array([0.]), nArr)
    etaArrNoGS = np.sqrt(.4 / nArr[1:]) / np.sqrt(prms.chainLength)
    etaArr = np.append(np.array([2.5 / np.sqrt(prms.chainLength)]), etaArrNoGS)

    print('nArr = {}'.format(nArr))

    print('etaArr = {}'.format(etaArr * np.sqrt(prms.chainLength)))
    assert(len(etaArr) == len(nArr))

    #gfArr = np.zeros((len(etaArr), len(kVec), len(wVec)),dtype=complex)
    #gfFloq = np.zeros((len(kVec), len(wVec)),dtype=complex) + 1e-5

    #gfArr = gfNonEqMultiParam.nonEqGreenMultiParamMultiProc(kPoint, wVec, damping, nArr)
    #gfArr = gfNonEqMultiParam.nonEqGreenMultiParamMultiProcAboveFS(kPoint, wVec, damping, nArr)
    #writeGreenToFile.writeGreen("data/nonEqGreenEtaAFSMany", "gfNonEq", gfArr)
    #writeGreenToFile.writeGreen("data/nonEqGreenEtaMany", "gfNonEq", gfArr)


    #tau = 2. * np.pi / prms.w0
    #tauLength = 1.
    #tAv = np.linspace(1. * tau, (1. + tauLength) * tau, 20, endpoint=False)
    #gWFloquet = floquetKArr.floquetGreenMultiProc(kVec, wVec, tAv, etaArr[-1], damping, nArr[-1])
    #gWFloquetInt = 1. / (tauLength * tau) * (tAv[1] - tAv[0]) * np.sum(gWFloquet, axis=2)
    #gfFloq = gWFloquetInt
    #writeGreenToFile.writeGreen("data/floquetGreen", "gfFloquet", gfFloq)

    gfFloq = readGreenFromFile.readGreen("data/floquetGreen", "gfFloquet")
    #gfFloqAFS = readGreenFromFile.readGreen("data/floquetGreenAFS", "gfFloquet")
    gfArr = readGreenFromFile.readGreen("data/nonEqGreenEtaMany", "gfNonEq")
    #gfArrAFS = readGreenFromFile.readGreen("data/nonEqGreenEtaAFS", "gfNonEq")
    #print("gfFloquet.shape = {}".format(gfFloq.shape))
    print("gfArr.shape = {}".format(gfArr.shape))
    bPlots.quantumToFloquetCrossover(wVec, 1. / np.sqrt(2. * np.pi) * gfArr, 1. / np.sqrt(2. * np.pi) * gfFloq[0, :], etaArr, nArr)
    #bPlots.quantumToFloquetCrossoverAFS(wVec, -1. / np.sqrt(2. * np.pi) * gfArrAFS, 1. / np.sqrt(2. * np.pi) * gfFloqAFS[0, :], etaArr, nArr)
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

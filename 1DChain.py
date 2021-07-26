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

from GiacomosPlot import gsSqueezing

def main():
    print('The length of the to-be-considered 1D chain is {}'.format(prms.chainLength))


#    ### Fig 1a
#    beuatifulPlots.plotFabry()
#
#    ### Fig 1b
#
#    prms.chainLength = 1010
#    prms.numberElectrons = prms.chainLength//2
#    etas = np.linspace(0., 2., 7, endpoint=True) / np.sqrt(prms.chainLength)
#    etas[3] = np.sqrt((np.pi) / (4. * prms.chainLength)) + 0.001 / np.sqrt(prms.chainLength)
#    beuatifulPlots.plotLandscapesAllOrders(etas, 3)
#    #Insets to signify the shifts
#    beuatifulPlots.plotShiftInsetes()
#
#    ###Fig 1c
#    origL = prms.chainLength
#    prms.chainLength = 510
#    etaLong = 2. / np.sqrt(prms.chainLength)
#    prms.numberElectrons = prms.chainLength // 2
#    gs = np.zeros((prms.chainLength))
#    gs[0: prms.numberElectrons // 2 + 1] = 1.
#    gs[- prms.numberElectrons // 2 + 1:] = 1.
#    gsJ = eF.J(gs)
#    gsTLong = eF.T(gs)
#    print("gsJ = {}".format(gsJ))
#    phGSL = phState.findPhotonGS([gsTLong, gsJ], etaLong, 3)
#
#    prms.chainLength = 10
#    eta = 2. / np.sqrt(prms.chainLength)
#    prms.numberElectrons = prms.chainLength // 2
#    gs = np.zeros((prms.chainLength))
#    gs[0: prms.numberElectrons // 2 + 1] = 1.
#    gs[- prms.numberElectrons // 2 + 1:] = 1.
#    gsJ = eF.J(gs)
#    gsT = eF.T(gs)
#    print("gsJ = {}".format(gsJ))
#    phGSS = phState.findPhotonGS([gsT, gsJ], eta, 3)
#
#    bPlots.plotPtGSWithCoh(phGSL, phGSS, etaLong, gsTLong)
#
#    prms.chainLength = origL
#
#    ### Fig 1d
#    gsSqueezing.callGiacomosCode()
#
#    prms.chainLength = 1010
#    prms.numberElectrons = prms.chainLength//2

#    ### Fig 2
#    etas = np.linspace(0., 2., 7, endpoint=True) / np.sqrt(prms.chainLength)
#    etas[3] = np.sqrt((np.pi) / (4. * prms.chainLength)) + 0.001 / np.sqrt(prms.chainLength)
#    beuatifulPlots.plotLandscapes1Order(etas, 1)
#    beuatifulPlots.plotLandscapes2Order(etas, 2)
#
#
    #calculate Green's function

#    prms.chainLength = 170
#    prms.maxPhotonNumber = 20
#    prms.numberElectrons = prms.chainLength//2

####Fig 3 a
#    damping = 0.025
#    eta = 1. / np.sqrt(prms.chainLength)
#    kVec = np.linspace(0, 2. * np.pi, prms.chainLength, endpoint=False)
#    wVec = np.linspace(-8, 8, 8000, endpoint=False)
#    #gAna2W = greenAna2nd.anaGreenVecW(kVec, wVec, eta, damping)
#    gfNumInf = greenNumArb.numGreenVecWGreater(kVec, wVec, eta, damping) + greenNumArb.numGreenVecWLesser(kVec, wVec, eta, damping)
#    GF = gfNumInf
#    #writeGreenToFile.writeGreen("data/eqGreenNum.h5", "gfEq", GF)
#    #GF = readGreenFromFile.readGreen("data/eqGreenNum.h5", "gfEq")
#    bPlots.plotSpecLogDashed(wVec, 1. / np.sqrt(2. * np.pi) * np.imag(np.transpose(GF)), eta)


    ####### Fig 3b
    prms.chainLength = 90
    prms.maxPhotonNumber = 60
    prms.numberElectrons = prms.chainLength//2

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

    print('nArr = {}'.format(nArr))

    print('etaArr = {}'.format(etaArr * np.sqrt(prms.chainLength)))
    assert(len(etaArr) == len(nArr))

    gfArr = gfNonEqMultiParam.nonEqGreenMultiParamMultiProc(kPoint, wVec, damping, nArr)
    #writeGreenToFile.writeGreen("data/nonEqGreenEtaMany", "gfNonEq", gfArr)

    tau = 2. * np.pi / prms.w0
    tAv = np.linspace(- .5 * tau, .5  * tau, 20, endpoint=False)
    gWFloquet = floquetKArr.floquetGreenMultiProc(kVec, wVec, tAv, etaArr[-1], damping, nArr[-1])
    gWFloquetInt = 1. / tau * (tAv[1] - tAv[0]) * np.sum(gWFloquet, axis=2)
    gfFloq = gWFloquetInt
    #writeGreenToFile.writeGreen("data/floquetGreen", "gfFloquet", gfFloq)

    #gfFloq = readGreenFromFile.readGreen("data/floquetGreen", "gfFloquet")
    #gfArr = readGreenFromFile.readGreen("data/nonEqGreenEtaMany", "gfNonEq")
    #print("gfArr.shape = {}".format(gfArr.shape))
    bPlots.quantumToFloquetCrossover(wVec, 1. / np.sqrt(2. * np.pi) * gfArr, 1. / np.sqrt(2. * np.pi) * gfFloq[0, :], etaArr, nArr)

    exit()

    #### Fig 4
    eta1 = 1. / np.sqrt(prms.chainLength)
    eta2 = 0.3 / np.sqrt(prms.chainLength)
    bPlots.plotAnalyticalConductivity(eta1, eta2, 0.)
    bPlots.plotAnalyticalConductivityImaginary(eta1, eta2, 0.)


    print("")
    print("The calculation has finished - Juhu!")

main()

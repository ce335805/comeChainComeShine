import globalSystemParams as prms
import numpy as np
import comparisonPlots as compPlot
from automatedTests import gfTests
from automatedTests import ftTests
from automatedTests import gsTests
from automatedTests import gsIsEigenstate
from automatedTests import nonEqTests
from automatedTests import floquetTests
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
from multiProcGreen import greenKArr
from multiProcGreen import floquetKArr
from fileHandling import writeGreenToFile
from fileHandling import readGreenFromFile

def main():
    print('The length of the to-be-considered 1D chain is {}'.format(prms.chainLength))

    #gfTests.runAllTests()
    #ftTests.runAllTests()
    #gsTests.runAllTests()
    #gsIsEigenstate.runAllTests()
    #nonEqTests.runAllTests()
    #floquetTests.runAllTests()

    #exit()

    #eta = 3. / np.sqrt(prms.chainLength)
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


    tau = 2. * np.pi / prms.w0
    wVec = np.linspace(-4., 4., 2000, endpoint=False)
    tAv = np.linspace(0. * tau, 5. * tau, 100, endpoint=False)
    kVec = np.linspace(-np.pi, np.pi, 17, endpoint=True)
    damping = .01

    LArr = np.array([102, 102])

    #gfFloq = readGreenFromFile.readGreen("data/floquetGreen", "gfFloquet")
    #gfArr = readGreenFromFile.readGreen("data/nonEqGreen", "gfNonEq")
    #bPlots.greenWaterFall(kVec, wVec, gfArr, LArr, gfFloq)
    #exit()

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


    print("")
    print("The calculation has finished - Juhu!")

main()


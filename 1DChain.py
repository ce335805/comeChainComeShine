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

    #bPlots.calculateAndPlotShakeOffs()

    #exit()

    cohN = .1
    eta = 2. / np.sqrt(prms.chainLength + 1)

    tau = 2. * np.pi / prms.w0
    wVec = np.linspace(-10., 10., 500, endpoint=False)
    tAv = np.linspace(0. * tau, 20. * tau, 100, endpoint=False)
    kVec = np.linspace(-np.pi, np.pi, 17, endpoint=True)
    damping = .1

    LArr = np.array([12, 52, 102])

    gfFloq = readGreenFromFile.readGreen("data/floquetGreen", "gfFloquet")
    gfArr = readGreenFromFile.readGreen("data/nonEqGreen", "gfNonEq")
    bPlots.greenWaterFall(kVec, wVec, gfArr, LArr, gfFloq)
    exit()

    gfArr = np.zeros((len(LArr), len(kVec), len(wVec)),dtype=complex)
    gfFloq = np.zeros((len(kVec), len(wVec)),dtype=complex)
    for lInd, lVal in enumerate(LArr):
        prms.chainLength = lVal
        prms.numberElectrons = lVal // 2
        cohN = lVal / 100.
        eta = 1. / np.sqrt(lVal)
        prms.maxPhotonNumber = int(20 + cohN)

        gfNonEq = greenKArr.nonEqGreenMultiProc(kVec, wVec, tAv, eta, damping, cohN)
        gfNonEqN0 = 1. / (21. * tau) * (tAv[1] - tAv[0]) * np.sum(gfNonEq, axis=2)

        gfArr[lInd, :, :] = gfNonEqN0

        if(lInd == len(LArr) - 1):
            gWFloquet = floquetKArr.floquetGreenMultiProc(kVec, wVec, tAv, eta, damping, cohN)
            gWFloquetInt = 1. / (21. * tau) * (tAv[1] - tAv[0]) * np.sum(gWFloquet, axis=2)
            gfFloq = gWFloquetInt


    writeGreenToFile.writeGreen("data/floquetGreen", "gfFloquet", gfFloq)
    writeGreenToFile.writeGreen("data/nonEqGreen", "gfNonEq", gfArr)



    print("")
    print("The calculation has finished - Juhu!")

main()


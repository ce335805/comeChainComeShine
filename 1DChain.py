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

def main():
    print('The length of the to-be-considered 1D chain is {}'.format(prms.chainLength))

    #gfTests.runAllTests()
    #ftTests.runAllTests()
    #gsTests.runAllTests()
    #gsIsEigenstate.runAllTests()
    nonEqTests.runAllTests()


    #tArr = np.linspace(-10., -10000., 201)
    #eta = 1.
    #diagonalizeH.plotPtOcc(tArr, eta)

    #etaArr = np.linspace(0., 1., 11)
    #lArr = np.array([10, 50, 100, 200])
    #photonOccupancies.plotPhotonOcc(lArr, etaArr, 2)

    #etas = np.linspace(0., 1., 11) / np.sqrt(prms.chainLength)
    #gsFromFSShift.plotLandscapes(etas, 2)

    #exit()

    #eta = .1
    #etas = np.linspace(0., 2., 21)
    #phNumbers = arbOrder.findPhotonNumbers(etas, 3)
    #plt.plot(etas, phNumbers)
    #plt.show()

    #gsE = arbOrder.findGS(eta, 3)
    #energyGS = photonState.energyFromState(gsE, eta, 3)
    #print("GS energy = {:.3f}".format(energyGS))

    #plt.plot(gsE)
    #plt.show()

    #jGauge = current.currentGS(eta)
    #print("jGS = {}".format(jGauge))

    #ptGS = arbOrder.getPhotonGS(eta, 3)
    #eGS = arbOrder.findGS(eta, 3)
    #N = photonState.avPhotonNum(eGS, eta, 3)
    #print("N_pt = {}".format(N))

    #bPlots.plotPtGSWithCoh(ptGS, N, eta)

    #exit()

    #eta = 2. / np.sqrt(prms.chainLength)

    #damping = .1
    #kVec = np.linspace(0, 2. * np.pi, prms.chainLength, endpoint=False)
    #wVec = np.linspace(-100., 100., 20000, endpoint=False)
    #A0 = 1.

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


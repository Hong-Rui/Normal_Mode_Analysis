from simtk.unit import *
from simtk.openmm import *
from simtk.openmm.app import *
import numpy as np
from scipy.interpolate import CubicSpline
from scipy.constants import pi, Boltzmann, hbar, Avogadro
from math import pi
from copy import deepcopy 
from warnings import warn
import matplotlib
import matplotlib.pyplot as plt
matplotlib.rcParams['figure.dpi'] = 600

class NormalModeAnalysis(object):
    """
        Create a NormalModeAnalysis object

        Args:
            topology (simtk.openmm.openmm.Topology): The OpenMM topology object
            system (simtk.openmm.openmm.System): The OpenMM system object.
            integrator (simtk.openmm.openmm.Integrator): The OpenMM integrator object.
            initPositions (numpy.array): The N*3 array of positions (N is the number of atoms).
            CPUProp (dictionary=None): The CPU platform-specific properties used for generating simulation object in OpenMM.
            CUDAProp (dictionary=None): The CUDA platform-specific properties used for generating simulation object in OpenMM.
    """
    def __init__(self, topology, system, integrator, initPositions, CPUProp=None, CUDAProp=None):
        self.topology = topology
        self.CPUSystem = deepcopy(system)
        self.CUDASystem = deepcopy(system)
        self.CPUIntegrator = deepcopy(integrator)
        self.CUDAIntegrator = deepcopy(integrator)
        self.initPositions = initPositions
        self.CPUProp = CPUProp
        self.CUDAProp = CUDAProp
        self.CPUSimulation = self.__getCPUSimulation__()
        self.CUDASimulation = self.__getCUDASimulation__()
        self.CPUSimulation.context.setPositions(self.initPositions)
        self.CUDASimulation.context.setPositions(self.initPositions)

    def __getDefaultCPUProperty__(self):
        pass

    def __getDefaultCUDAProperty__(self):
        return {'CudaDeviceIndex': '0', 'CudaPrecision': 'double', 'DeterministicForces': 'true'}

    def __getCPUSimulation__(self):
        if self.CPUProp:
            CPUPlatform = Platform.getPlatformByName('CPU')
            simulation = Simulation(self.topology, self.CPUSystem, self.CPUIntegrator, CPUPlatform, self.CPUProp)
            return simulation
        else:
            CPUPlatform = Platform.getPlatformByName('CPU')
            simulation = Simulation(self.topology, self.CPUSystem, self.CPUIntegrator, CPUPlatform)
            return simulation

    def __getCUDASimulation__(self):
        if self.CUDAProp:
            CUDAPlatform = Platform.getPlatformByName('CUDA')
            simulation = Simulation(self.topology, self.CUDASystem, self.CUDAIntegrator, CUDAPlatform, self.CUDAProp)
            return simulation
        else:
            CUDAPlatform = Platform.getPlatformByName('CUDA')
            self.CUDAProp = self.__getDefaultCUDAProperty__()
            simulation = Simulation(self.topology, self.CUDASystem, self.CUDAIntegrator, CUDAPlatform, self.CUDAProp)
            return simulation

    def __getVibrationalSpectrum__(self, SquareAngularFreq):
        SquareAngularFreqSI = (4.184*10**26)*SquareAngularFreq
        AngularFreqSI = np.sqrt(SquareAngularFreqSI)
        VibrationalSpectrum = AngularFreqSI/(6*pi*10**10)
        return VibrationalSpectrum

    def __checkSymmetric__(self, array2D, tol=1e-8):
        return np.allclose(array2D, array2D.T, atol=tol)

    def __checkPositiveDefinite__(self, eigVals):
        eigVals = eigVals[6:]
        return np.all(eigVals > 0)

    def CPUPreMinimization(self):
        """
            Initial minimization in CPU platform to remove bad contacts of the initial input positions.
            This function uses all the default settings of the class method simtk.openmm.app.simulation.Simulation.minimizeEnergy(). 
            After the initial minimization, set the self.CUDASimulation.context to the pre-minimized state.
            The mean force after minimization would fall in around the scale of ~ 0.2 kcal/(A mol).
        """
        self.CPUSimulation.minimizeEnergy()
        PreMinimizedState = self.CPUSimulation.context.getState(getPositions=True, getEnergy=True, getForces=True)
        self.CUDASimulation.context.setState(PreMinimizedState)

    def CUDAMinimizationCycle(self, MiniTolerance=0, MaxMiniCycle=1000, NumMiniStepPerCycle=10000, MiniForceRatio=1e-6):
        """
            Designed energy minimization cycle to minimize the structure such that the system mean force would fall around 2e-07 kcal/(A mol).
            This function will use the default positions to minimize.
            If the user did not to self.CPUPreMinimization() first, then the initial input positions will be used.
            Otherwise the pre-minimized positions will be used for performing the minimization cycle in CUDA platform.

            Args:
            MiniTolerance (energy=0*kilojoule/mole): The energy tolerance to which the system should be minimized set for each cycle.
            MaxMiniCycle (int=1000): The maximum number of cycles to perform energy minimizations.
            NumMiniStepPerCycle (int=10000): MaxIterations for each cycle of energy minimization.
            MiniForceRatio (double=1e-6): The order of mean force that the minimization cycle should eliminated.
        """
        PreMinimizedState = self.CUDASimulation.context.getState(getPositions=True, getEnergy=True, getForces=True)
        PreMinimizedForces = PreMinimizedState.getForces(asNumpy=True).value_in_unit(kilocalorie/(mole*angstrom))
        PreMinimizedMeanForce = np.linalg.norm(PreMinimizedForces,axis=1).mean() * (kilocalorie/(mole*angstrom))
        self.MiniForceThreshold = PreMinimizedMeanForce * MiniForceRatio

        for i in range(MaxMiniCycle):
            self.CUDASimulation.minimizeEnergy(tolerance=MiniTolerance*kilojoule/mole, maxIterations=NumMiniStepPerCycle)
            currentState = self.CUDASimulation.context.getState(getForces=True)
            currentForces = currentState.getForces(asNumpy=True).value_in_unit(kilocalorie/(mole*angstrom))
            currentMeanForce = np.linalg.norm(currentForces,axis=1).mean() * (kilocalorie/(mole*angstrom))
            if currentMeanForce < self.MiniForceThreshold:
                break

    def CalculateNormalModes(self, TweakEnergyRatio=1e-12):
        """
            The core function to do Quasi-Harmonic Analysis.
            This function takes the state from self.CUDASimulation.context.getState() as the minimized/equilibrium state.
            It is highly recommended that users perform self.CPUPreMinimization() and self.CUDAMinimizationCycle() before calculating normal modes.
            The algorithm is the following: 
            1. Tweak positions for each 3N positional dimensions at all 6N directions and calculate the corresponding N*3 force array, respectively.
            2. For each of the 3N dimension, use the cubic spline function to calculate first derivatives of the tweaked forces.
            3. Form a 3N*3N spring constant matrix and de-mass-weight it to make its unit to be square of angular frequency.
            4. Do eigenvalue decomposition to obtian normal modes and the corresponding eivenvalues representing the characteristic angular frequencies of that modes.
            5. Calculate the vibrational power spectrum from the eigenvalues.

            Args:
            TweakEnergyRatio (double=1e-12): The pre-designed order of the change of the potential energy to determine positional tweak displacements.
        """
        MinimizedState = self.CUDASimulation.context.getState(getPositions=True, getEnergy=True, getForces=True)
        MinimizedPositions = MinimizedState.getPositions(asNumpy=True).in_units_of(angstrom)
        MinimizedPotentialEnergy = MinimizedState.getPotentialEnergy().in_units_of(kilocalorie/mole)
        MinimizedForces = MinimizedState.getForces(asNumpy=True).value_in_unit(kilocalorie/(mole*angstrom))
        MinimizedMeanForce = np.linalg.norm(MinimizedForces,axis=1).mean() * (kilocalorie/(mole*angstrom))

        NumAtoms = self.CUDASimulation.system.getNumParticles()
        TweakEnergyDiff = MinimizedPotentialEnergy/NumAtoms * TweakEnergyRatio
        self.TweakDisplacement = abs((TweakEnergyDiff/MinimizedMeanForce).in_units_of(angstrom))
        NumDimension = MinimizedForces.shape[0]*MinimizedForces.shape[1]
        Forces3N = MinimizedForces.reshape((1,NumDimension))
        Positions3N = MinimizedPositions.reshape((1,NumDimension))

        TweakEnergies = np.zeros((6*NumAtoms,1))
        MeanForces = np.zeros((6*NumAtoms,1))
        TweakForces = np.zeros((6*NumAtoms,NumDimension))
        SpringConsts = np.zeros((NumDimension,NumDimension))
        ForcesZero = np.copy(Forces3N[0])

        for i in range(NumDimension):
            currentPositions3NPos = np.copy(Positions3N[0])
            currentPositions3NPos[i] += self.TweakDisplacement.value_in_unit(angstrom)
            currentPositions3NPosQuantity = currentPositions3NPos.reshape((NumAtoms,3)) * angstrom
            currentPositions3NPosQuantity = currentPositions3NPosQuantity.in_units_of(nanometer)
            self.CUDASimulation.context.setPositions(currentPositions3NPosQuantity)
            NewStatePos = self.CUDASimulation.context.getState(getEnergy=True, getForces=True)
            NewForcesPos = NewStatePos.getForces(asNumpy=True).value_in_unit(kilocalorie/(mole*angstrom)).reshape((1,NumDimension))[0]
            TweakForces[2*i,:] = NewForcesPos
            TweakEnergies[2*i] = NewStatePos.getPotentialEnergy().value_in_unit(kilocalorie/mole)
            MeanForces[2*i] = np.linalg.norm(NewForcesPos.reshape((NumAtoms,3)),axis=1).mean()
    
            currentPositions3NNeg = np.copy(Positions3N[0])
            currentPositions3NNeg[i] -= self.TweakDisplacement.value_in_unit(angstrom)
            currentPositions3NNegQuantity = currentPositions3NNeg.reshape((NumAtoms,3)) * angstrom
            currentPositions3NNegQuantity = currentPositions3NNegQuantity.in_units_of(nanometer)
            self.CUDASimulation.context.setPositions(currentPositions3NNegQuantity)
            NewStateNeg = self.CUDASimulation.context.getState(getEnergy=True, getForces=True)
            NewForcesNeg = NewStateNeg.getForces(asNumpy=True).value_in_unit(kilocalorie/(mole*angstrom)).reshape((1,NumDimension))[0]
            TweakForces[2*i+1,:] = NewForcesNeg
            TweakEnergies[2*i+1] = NewStateNeg.getPotentialEnergy().value_in_unit(kilocalorie/mole)
            MeanForces[2*i+1] = np.linalg.norm(NewForcesNeg.reshape((NumAtoms,3)),axis=1).mean()
    
            PosVariable = np.array([-self.TweakDisplacement.value_in_unit(angstrom), 0, self.TweakDisplacement.value_in_unit(angstrom)])
            ForceVariable = np.array([NewForcesNeg,ForcesZero,NewForcesPos]).T
            ForceVariable = ForceVariable*(-1)
            ForceFunction = CubicSpline(PosVariable,ForceVariable,axis=1)
            SpringConsts[i,:] = ForceFunction(0,1)

        MassArray = np.zeros(NumAtoms)
        for i in range(NumAtoms):
            MassArray[i] = self.CUDASimulation.system.getParticleMass(i).value_in_unit(dalton)

        MassArray3N = np.sqrt(MassArray.repeat(3))
        MassMatrix = np.outer(MassArray3N,MassArray3N)
        Hessian = np.divide(SpringConsts,MassMatrix)
        HessianSymmetric = np.mean([Hessian,Hessian.T],axis=0)
        eigVal, eigVec = np.linalg.eig(HessianSymmetric)
        sortIdx = np.argsort(eigVal)
        eigValSorted = eigVal[sortIdx]
        eigVecSorted = eigVec[:,sortIdx]
        VibrationalSpectrum = self.__getVibrationalSpectrum__(eigValSorted[6:])

        if not self.__checkSymmetric__(HessianSymmetric):
            warn('Fatal Warining: The hessian is NOT symmetric !!')

        if not self.__checkPositiveDefinite__(eigValSorted):
            warn('Fatal Warning: The hessian is NOT positive definite !!')

        self.Hessian = HessianSymmetric * (kilocalorie/(gram*angstrom**2))
        self.SquareAngularFreq = eigValSorted * (kilocalorie/(gram*angstrom**2))
        self.NormalModes = eigVecSorted
        self.VibrationalSpectrum = VibrationalSpectrum * (1/centimeter)

    def PlotVibrationalSpectrum(self, binNum=1000, colorStr='salmon', labelStr='Vibrational Power Spectrum'):
        """
            Plot the histogram of vibrational power spectrum intensities.
            X-axis (cm^-1): Wave number.
            Y-axis (dimensionless number): Intensity of the histogram.
            Args:
            binNum (int=1000): Number of bins to generate the histogram.
            colorStr (string='salmon'): The matplotlib color string.`
            labelStr (string='Vibrational Power Spectrum'): The string used to show the label of current histogram. 
        """
        matplotlib.rcParams['figure.dpi'] = 600
        fig, ax = plt.subplots()
        ax.hist(self.VibrationalSpectrum, bins=binNum, color=colorStr,label=labelStr)
        ax.legend()
        plt.xlabel(r'Wave number($cm^{-1}$)')
        plt.ylabel(r'Intensity')
        plt.xlim(0,4000)
        plt.show()

    def getVibrationalEntropyCM(self, Temperature=300*kelvin):
        SquareAngularFreqAKMA = self.SquareAngularFreq.value_in_unit(kilocalorie/(gram*angstrom**2))[6:]
        SquareAngularFreqSI = (4.184*10**26)*SquareAngularFreqAKMA
        NumAtoms = self.CUDASimulation.system.getNumParticles()
        internalDim = 3*NumAtoms - 6
        Temperature = Temperature.value_in_unit(kelvin)
        kBT = Boltzmann*Temperature
        VibrationalEntropyCM = (internalDim*kBT/2) * (1 + np.log(2*pi*kBT)) - (kBT/2) * (np.sum(np.log(SquareAngularFreqSI)))
        VibrationalEntropyCM = VibrationalEntropyCM*Avogadro * (joule/mole)
        self.VibrationalEntropyCM = VibrationalEntropyCM.in_units_of(kilocalorie/mole)
    
    def getVibrationalEntropyQM(self, Temperature=300*kelvin):
        SquareAngularFreqAKMA = self.SquareAngularFreq.value_in_unit(kilocalorie/(gram*angstrom**2))[6:]
        AngularFreqSI = np.sqrt((4.184*10**26)*SquareAngularFreqAKMA)
        NumAtoms = self.CUDASimulation.system.getNumParticles()
        internalDim = 3*NumAtoms - 6
        Temperature = Temperature.value_in_unit(kelvin)
        kBT = Boltzmann*Temperature
        quantumEnergy = hbar*AngularFreqSI
        VibrationalEntropyQMArray = quantumEnergy/(np.exp(quantumEnergy/kBT) - 1) - kBT*np.log(1 - np.exp(-quantumEnergy/kBT))
        VibrationalEntropyQM = np.sum(VibrationalEntropyQMArray)
        VibrationalEntropyQM = VibrationalEntropyQM*Avogadro * (joule/mole)
        self.VibrationalEntropyQM = VibrationalEntropyQM.in_units_of(kilocalorie/mole)



        

